import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

def HardwareIdentification():
    if torch.cuda.is_available():
        Device = "cuda"
    else:
        Device = "cpu"
    print(Device)
    return Device

#Training function
def Trainloop(DataLoader, Model, Loss, Optimizer, device):
    Sum = 0
    #For all the batched from the loader
    for Iter, (X, y) in enumerate(DataLoader):
        #It is required to compute the gradient
        X.requires_grad = True
        #Model prediction
        y_pred = Model(X.to(device))
        #Loss computation
        Myloss = Loss(y_pred, y.to(device))
        #Backward propagation
        Optimizer.zero_grad()
        Myloss.backward()
        Optimizer.step()

        Sum += Myloss.item()
    Sum /= len(DataLoader)

    #Return the loss
    return Sum

def TestLoop(TestLoader, Model, device):
    Model.eval()  

    correct = 0
    total = 0

    with torch.no_grad(): 
        for data in TestLoader:
            images, labels = data  
            outputs = Model(images.to(device)) 
            
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item() 

    accuracy = 100 * correct / total  
    print(f'Accuracy: {accuracy:.2f}%')
    
def AddSecurity(Image):
    Image2 = Image.clone()
    Image2 = (Image * 10).int().float() / 10
    return Image2
    
def CalculateNorms(pre, Ppred, Apred, Spre, MC, MD, Prob, Activation, c, Coef, S):
    MDiff = MC.to("cpu") - MD.to("cpu")
    
    l0_norm = torch.nonzero(MDiff).size(0)
    l2_norm = torch.norm(MDiff, p=2).numpy()
    linfty_norm = torch.norm(MDiff, p=float('inf')).numpy()

    print(l0_norm, l2_norm, linfty_norm, Activation, Coef, c, Prob, S, pre.numpy()[0], Ppred.numpy()[0], Apred.numpy()[0], Spre.numpy()[0])
    return [l0_norm, l2_norm, linfty_norm, Activation, Coef, Prob, c, S, pre.numpy()[0], Ppred.numpy()[0], Apred.numpy()[0], Spre.numpy()[0]]
    
def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()