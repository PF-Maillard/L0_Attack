import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result
    return wrapper_func

class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """

    def __init__(self, name, model, device):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        if device:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = None
                print(
                    'Failed to set device automatically, please try set_device() manual.')

        # Controls attack mode.
        self.attack_mode = 'default'
        self.supported_mode = ['default']
        self.targeted = False
        self._target_map_function = None

        # Controls when normalization is used.
        self.normalization_used = {}
        self._normalization_applied = False
        self._set_auto_normalization_used(model)

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def _check_inputs(self, images):
        tol = 1e-4
        if self._normalization_applied:
            images = self.inverse_normalize(images)
        if torch.max(images) > 1+tol or torch.min(images) < 0-tol:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
        return images

    def _check_outputs(self, images):
        if self._normalization_applied:
            images = self.normalize(images)
        return images

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_auto_normalization_used(self, model):
        if model.__class__.__name__ == 'RobModel':
            mean = getattr(model, 'mean', None)
            std = getattr(model, 'std', None)
            if (mean is not None) and (std is not None):
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                if (mean != 0).all() or (std != 1).all():
                    self.set_normalization_used(mean, std)
    #                 logging.info("Normalization automatically loaded from `model.mean` and `model.std`.")

    @wrapper_method
    def set_normalization_used(self, mean, std):
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs*std + mean

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self.attack_mode = 'default'
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted('targeted(custom)', quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        r"""
        Set attack mode as targeted with random labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted('targeted(random)', quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        r"""
        Set attack mode as targeted with least likely labels.

        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            num_classses (str): number of classes. (Default: False)

        """
        self._set_mode_targeted('targeted(least-likely)', quiet)
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        .. note::
            Use user-supplied labels as target labels.
        """
        self._set_mode_targeted('targeted(label)', quiet)
        self._target_map_function = 'function is a string'

    @wrapper_method
    def set_model_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False,
             save_predictions=False, save_clean_inputs=False, save_type='float'):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(batch_size, -1)  # nopep8
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step+1)/total_batch*100
                    end = time.time()
                    elapsed_time = end-start

                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')  # nopep8

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {'adv_inputs': adv_input_list_cat, 'labels': label_list_cat}  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict['preds'] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict['clean_inputs'] = input_list_cat

                if self.normalization_used is not None:
                    save_dict['adv_inputs'] = self.inverse_normalize(save_dict['adv_inputs'])  # nopep8
                    if save_clean_inputs:
                        save_dict['clean_inputs'] = self.inverse_normalize(save_dict['clean_inputs'])  # nopep8

                if save_type == 'int':
                    save_dict['adv_inputs'] = self.to_type(save_dict['adv_inputs'], 'int')  # nopep8
                    if save_clean_inputs:
                        save_dict['clean_inputs'] = self.to_type(save_dict['clean_inputs'], 'int')  # nopep8

                save_dict['save_type'] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        r"""
        Return inputs as int if float is given.
        """
        if type == 'int':
            if isinstance(inputs, torch.FloatTensor) or isinstance(inputs, torch.cuda.FloatTensor):
                return (inputs*255).type(torch.uint8)
        elif type == 'float':
            if isinstance(inputs, torch.ByteTensor) or isinstance(inputs, torch.cuda.ByteTensor):
                return inputs.float()/255
        else:
            raise ValueError(
                type + " is not a valid type. [Options: float, int]")
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t'
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @staticmethod
    def load(load_path, batch_size=128, shuffle=False, normalize=None,
             load_predictions=False, load_clean_inputs=False):
        save_dict = torch.load(load_path)
        keys = ['adv_inputs', 'labels']

        if load_predictions:
            keys.append('preds')
        if load_clean_inputs:
            keys.append('clean_inputs')

        if save_dict['save_type'] == 'int':
            save_dict['adv_inputs'] = save_dict['adv_inputs'].float()/255
            if load_clean_inputs:
                save_dict['clean_inputs'] = save_dict['clean_inputs'].float() / 255  # nopep8

        if normalize is not None:
            n_channels = len(normalize['mean'])
            mean = torch.tensor(normalize['mean']).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize['std']).reshape(1, n_channels, 1, 1)
            save_dict['adv_inputs'] = (save_dict['adv_inputs'] - mean) / std
            if load_clean_inputs:
                save_dict['clean_inputs'] = (save_dict['clean_inputs'] - mean) / std  # nopep8

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(
            adv_data, batch_size=batch_size, shuffle=shuffle)
        print("Data is loaded in the following order: [%s]" % (", ".join(keys)))  # nopep8
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError(
                'target_map_function is not initialized by set_mode_targeted.')
        if self.attack_mode == 'targeted(label)':
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, images, labels=None, *args, **kwargs):
        if self.device:
            given_training = self.model.training
            self._change_model_mode(given_training)
            images = self._check_inputs(images)
            adv_images = self.forward(images, labels, *args, **kwargs)
            adv_images = self._check_outputs(adv_images)
            self._recover_model_mode(given_training)
            return adv_images
        else:
            print('Device is not set.')

    def __repr__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack', 'supported_mode']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self.attack_mode
        info['normalization_used'] = True if len(self.normalization_used) > 0 else False  # nopep8

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get('_attacks')

        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if (items not in stack):
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = (list(items.keys())+list(items.values()))
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        for num, value in enumerate(get_all_values(value)):
            attacks[name+"."+str(num)] = value
            for subname, subvalue in value.__dict__.get('_attacks').items():
                attacks[name+"."+subname] = subvalue


def CalculatePixel(a,b,c, Image):
    Result = 0
    Nb=0
    Channels, Rows, Cols = Image.shape
    ReferencePoint = Image[a][b][c].item()

    if (b - 1) >= 0:
        A = math.sqrt((Image[a][b - 1][c].item() - ReferencePoint)**2)
        Result += A
        Nb +=1
    if (b + 1) < Rows:
        A = math.sqrt((Image[a][b + 1][c].item() - ReferencePoint)**2)
        Result += A
        Nb +=1
    if (c - 1) >= 0:
        A = math.sqrt((Image[a][b][c - 1].item() - ReferencePoint)**2)
        Result += A
        Nb +=1
    if (c + 1) < Cols:
        A = math.sqrt((Image[a][b][c + 1].item() - ReferencePoint)**2)
        Result += A
        Nb +=1
        
    return 1 - Result/Nb #1 - Result/Nb #  

def CoherentImage(Image):
  Coherent = torch.empty_like(Image)
  Channels, Rows, Cols = Image.shape
  for channel in range(Channels):
      for row in range(Rows):
          for col in range(Cols):
            Coherent[channel][row][col] = CalculatePixel(channel, row, col, Image)
  return Coherent



class PFM(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(self, model, device=None, c=1, kappa=0, steps=50, lr=0.01, Coef=10, Activation=0, Prob = 5, Similarity=0.1, Init=0):
        super().__init__('PFM', model, device)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ['default', 'targeted']
        self.Coef=Coef
        self.Activation=Activation
        self.Prob = Prob
        self.Similarity = Similarity
        self.Init = Init

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if self.Init == 0:
            w = self.inverse_tanh_space(images).detach()
        if self.Init == 1: 
            w = torch.zeros_like(images).detach()
        if self.Init == 2: 
            w = torch.rand_like(images).detach() 
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MyLoss = MyNorm(self.Coef, self.Activation)
        MSELoss = nn.MSELoss(reduction='none')

        optimizer = optim.Adam([w], lr=self.lr)
        #optimizer = optim.SGD([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            #adv_images = self.tanh_space(w)  

            adv_images2 = self.tanh_space(w.clone()) 
            token = 0
            #adv_images2 = adv_images
            #if (step % int(self.steps / 100)) == 0:

            if random.randint(0, self.Prob-1) == 0:
                adv_images2 = self.ImageClose(images, adv_images2, self.Similarity) 
                token = 1

            # Calculate loss
            current_L2 = MyLoss(images,adv_images2).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.get_logits(adv_images2)

            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)

            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            #print(cost.to("cpu").detach().numpy(), L2_loss.to("cpu").detach().numpy(), self.c * f_loss.to("cpu").detach().numpy(), pre[1])

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            if token == 1:
                mask1 = condition*(best_L2 > current_L2.detach())
                best_L2 = mask1*current_L2.detach() + (1-mask1)*best_L2

                mask = mask1.view([-1]+[1]*(dim-1))
                best_adv_images = mask*adv_images2.detach() + (1-mask)*best_adv_images
                #print(best_L2.to("cpu").numpy(), labels.to("cpu").numpy(), pre.to("cpu").numpy(), condition.to("cpu").numpy(), mask.to("cpu").numpy(), mask1.to("cpu").numpy())


            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10, 1) == 0:
                if cost.item() > prev_cost:
                    #best_adv_images = self.ImageClose(images, best_adv_images) 
                    return best_adv_images
                prev_cost = cost.item()
                
        #best_adv_images = self.ImageClose(images, best_adv_images) 
        return best_adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x*2-1, min=-1, max=1))

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def ImageClose(self, img, adv, Similarity):
        adv2 = adv.clone()
        diff = torch.abs(img - adv)
        
        mask = diff < Similarity
        
        adv2[mask] = img[mask]
        
        return adv2

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        Bug = (1-one_hot_labels)*outputs + one_hot_labels * (-1e9)
        other = torch.max(Bug, dim=1)[0]
        # get the target class's logit
        #real = torch.max(one_hot_labels*outputs, dim=1)[0]
        real = torch.max(one_hot_labels*outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other-real), min=-self.kappa)
        else:
            return torch.clamp((real-other), min=-self.kappa)
        

class MyNorm(nn.Module):
    def __init__(self, c, Activation):
        super(MyNorm, self).__init__()
        self.Flatten = nn.Flatten()
        self.c = c
        self.Activation = Activation
    
    def sigmoid(self, x, c ):
        return 1 / (1 + torch.exp(-x*c))

    def lure(self, x, c):
        return torch.min(x*c, torch.ones_like(x))

    def tanh(self, x, c):
        return torch.tanh(x*c)
    
    def relu(self, x, c):
        return torch.max(x*c-0.1, torch.zeros_like(x))
    
    def MyObjective(self, C, D):
        L = ((C - D)**2)
        Result = L
        
        if self.Activation == 1:
            Result = self.lure(L, self.c) 
        if self.Activation == 2:
            Result = self.sigmoid(L, self.c)
        if self.Activation == 3:
            Result = self.relu(L, self.c) 
        if self.Activation == 4:
            Result = self.tanh(L, self.c)
            
        return Result

    def forward(self, ListImage1, ListImage2):
        VectorList = []
        
        for i in range(len(ListImage1)):
            C = self.Flatten(ListImage1[i]).view(-1)
            D = self.Flatten(ListImage2[i]).view(-1)
            loss = self.MyObjective(C,D)
            VectorList.append(loss)
        TotalLoss = torch.stack(VectorList)
        return TotalLoss
