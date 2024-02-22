import time
import pandas as pd 
import torchattacks
from torchvision.utils import save_image

import Sources.Maillard
import Sources.DefaultModel as Util
import Sources.CW_L0


def SaveImages(MyPath, ImagesData, y):
    for i, tensor in enumerate(ImagesData):
        Path = MyPath + "image_" + str(i) + "_" + str(y[i]) + ".png"
        save_image(tensor, Path)

def SaveJSMA(theta, gamma, Model, X, y, Device, MyPath):
    Data = []
    MyAtk = torchattacks.JSMA(Model, theta=theta, gamma=gamma)
    AdvPFM = MyAtk(X, y)
    for idx in range(len(X)):
        TPred = y[idx:idx+1]
        Ppre = Util.get_pred(Model, X[idx:idx+1], Device)
        Apre = Util.get_pred(Model, AdvPFM[idx:idx+1], Device)
        adv2 = Util.AddSecurity(X[idx:idx+1])
        Spre = Util.get_pred(Model, adv2, Device) 
        Norm = Util.CalculateNorms(TPred, Ppre, Apre, Spre, X[idx:idx+1], AdvPFM[idx:idx+1], 0, 0, 0, 0, 0)
        Path = MyPath + "image_" + str(idx) + "_" + str(Norm[0]) + ".png"
        save_image(AdvPFM[idx:idx+1], Path)
    return [Data]

def AttackNorms(Model, C, steps, lr, Coef, Activation, P, S, Init, X, y, Device):
    Data = []
    Images = []
    atk = Maillard.PFM(Model, c=C, steps=steps, lr=lr, Coef=Coef, Activation=Activation, Prob = P, Similarity = S, Init=Init)
    AdvPFM = atk(X, y)
    for idx in range(len(X)):
        TPred = y[idx:idx+1]
        Ppre = Util.get_pred(Model, X[idx:idx+1], Device)
        Apre = Util.get_pred(Model, AdvPFM[idx:idx+1], Device)
        
        adv2 = Util.AddSecurity(X[idx:idx+1])
        Spre = Util.get_pred(Model, adv2, Device) 
        ToSave = Util.CalculateNorms(TPred, Ppre, Apre, Spre, X[idx:idx+1], AdvPFM[idx:idx+1], P, Activation, C, Coef, S)
        Data.append(ToSave)
        Images.append(AdvPFM[idx:idx+1])
    return Data, Images

def SaveTime(start_time, end_time, FileName):
    execution_time = end_time - start_time
    with open(FileName, 'w') as file:
        file.write(f'Execution time: {execution_time} seconds\n')
        
def SaveData(data, Name):
    Fdata = []
    for i in range(len(data[0])):
        min = 784
        ToSave = 0
        for j in range(len(data)):
            if data[j][i][8] != data[j][i][10] and data[j][i][0] < min:
                min = data[j][i][0]
                ToSave = j
        Fdata.append(data[ToSave][i])

    print(Fdata)
    df = pd.DataFrame(Fdata, columns=['L0 Norm', 'L2 Norm', 'Linf Norm', 'Activation', 'Coef', 'Prob', 'c', 'Similarity', 'Pre', 'Ppred', 'Apred', 'Spre'])
    df.to_excel(Name)    
        
def TestLoopC(Model, steps, lr, Coef, Activation, P, S, Init, X, y, Device, Start, End, Factor):
    data = []
    C = Start
    while(C > End):
        DataResult, _ = AttackNorms(Model, C, steps, lr, Coef, Activation, P, S, Init, X, y, Device)
        data.append(DataResult)
        C/=Factor
    return data

def TestJSMA(theta, gamma, Model, X, y, Device):
    Data = []
    MyAtk = torchattacks.JSMA(Model, theta=theta, gamma=gamma)
    AdvPFM = MyAtk(X, y)
    for idx in range(len(X)):
        TPred = y[idx:idx+1]
        Ppre = Util.get_pred(Model, X[idx:idx+1], Device)
        Apre = Util.get_pred(Model, AdvPFM[idx:idx+1], Device)
        
        adv2 = Util.AddSecurity(X[idx:idx+1])
        Spre = Util.get_pred(Model, adv2, Device) 
        ToSave = Util.CalculateNorms(TPred, Ppre, Apre, Spre, X[idx:idx+1], AdvPFM[idx:idx+1], 0, 0, 0, 0, 0)
        Data.append(ToSave)
    return [Data]

def TestCW(Iterations, InitialConst, MaxConst, Model, X, y, Device, num_channels = 1, image_size=28, DimOutput = 10):
    dataCW = []
    atk = CW_L0.CarliniL0(Model, num_labels = DimOutput, image_size=image_size, num_channels=num_channels, batch_size=1, max_iterations=Iterations, initial_const=InitialConst, largest_const=MaxConst)
    AdvCW = atk.attack(X.to(Device), y.to(Device), Device)     

    for idx in range(len(X)):     
        TPred = y[idx:idx+1]
        Ppre = Util.get_pred(Model, X[idx:idx+1], Device)
        Apre = Util.get_pred(Model, AdvCW[idx:idx+1], Device)
        adv2 = Util.AddSecurity(AdvCW[idx:idx+1])
        Spre = Util.get_pred(Model, adv2, Device) 
        ToSave = Util.CalculateNorms(TPred, Ppre, Apre, Spre, X[idx:idx+1].detach(), AdvCW[idx:idx+1].detach(), 0, 0, 0, 0, 0) 
        dataCW.append(ToSave)
    return dataCW, AdvCW

def SaveLoopC(Model, steps, lr, Coef, Activation, P, S, Init, X, y, Device, Start, End, Factor, MyPath):
    NormsData = []
    ImagesData = []
    C = Start
    while(C > End):
        Norms, Images = AttackNorms(Model, C, steps, lr, Coef, Activation, P, S, Init, X, y, Device)
        if NormsData == []:
            NormsData = Norms
            ImagesData = Images
        else:
            for i in range(len(Norms)):
                if Norms[i][8] != Norms[i][10] and Norms[i][0] < NormsData[i][0]:
                    NormsData[i] = Norms[i]
                    ImagesData[i] = Images[i]
        C/=Factor

    for i, tensor in enumerate(ImagesData):
        Path = MyPath + "image_" + str(i) + "_" + str(NormsData[i][0]) + ".png"
        save_image(tensor, Path)
        
def SaveImages(MyPath, ImagesData, y):
    for i, tensor in enumerate(ImagesData):
        Path = MyPath + "image_" + str(i) + "_" + str(y[i]) + ".png"
        save_image(tensor, Path)