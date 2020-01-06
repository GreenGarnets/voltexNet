import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import codecs

from torch.autograd import Variable
from dataset import KshDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from net.model import DeepLabV3
from PIL import Image
def targetToTensor(target):   
    output1 = torch.randint(1, size=(512,))
    output2 = torch.randint(1, size=(512,))
    target = target.replace("\n","")
    #print(target)
    #print(len(target))
    #print(output.shape)
    for i in range(0,len(target)):
        #print(target[i])
        if ((i+1)%8==7 or (i+1)%8==0) and i != 0:
            if target[i] == '-' :
                output2[i] = 0
                output1[i] = 0
            elif target[i] == 'o':
                output2[i] = 1
                output1[i] = 0
            elif target[i] == ':':
                output2[i] = 2
                output1[i] = 0
            elif target[i] == '0':
                output2[i] = 3
                output1[i] = 0
            elif target[i] == 'D':
                output2[i] = 4
                output1[i] = 0
            elif target[i] == 'c':
                output2[i] = 5
                output1[i] = 0
            elif target[i] == 'C':
                output2[i] = 6
                output1[i] = 0
            else:
                output2[i] = 7
        else :
            output2[i] = 0
            output1[i] = int(target[i])
    print("--")
    for i in range(0,512):
        print(output1[i].item(), end = ' ')
        if (i+1)%8 == 0: 
            print("")
    print("--")

    for i in range(0,512):
        print(output2[i].item(), end = ' ')
        if (i+1)%8 == 0: 
            print("")
    
    print("--")
    
    return output1, output2

if __name__ == "__main__":
    note_model = DeepLabV3(3)
    nove_model = DeepLabV3(8)
    note_model.load_state_dict(torch.load("./model/note_model.pth"))
    nove_model.load_state_dict(torch.load("./model/nove_model.pth"))

    CrossEntropyLoss = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # move model to the right device
    note_model.to(device)
    nove_model.to(device)

    dirname = "./infer/"
    #dirTargetname = "./infer/ksh_cut/"
    filenames = os.listdir(dirname)

    output = codecs.open("./infer/output.txt","w")

    for filename in filenames:
        print(filename)
        output.write(filename + "\n")
        img_path = os.path.join(dirname, filename)
        #ksh_path = os.path.join(dirTargetname, filename.replace(".png",".txt"))
        
        img = Image.open(img_path).convert("RGB")
        
        input = ToTensor()(img).unsqueeze(0)
        #input = input.squeeze()
        input = input.to(device)
        '''
        targetFile = codecs.open(ksh_path)
        target = targetFile.read()
        target1, target2 = targetToTensor(target)
        target1 = target1.to(device)
        target2 = target2.to(device)
        '''
        pred1 = note_model(input)
        pred2 = nove_model(input)
        #loss1 = CrossEntropyLoss(pred1, target1)
        #loss2 = CrossEntropyLoss(pred2, target2)

        #print(pred1.shape)
        #print(pred2.shape)

        outstr = ''
        for i in range(0,512) :
            max = -100
            maxNum = 0
            if ((i+1)%8 == 7 or (i+1)%8 == 0) and i != 0 :    
                for j in range(0,8) :
                    if int(pred2[i,j]) > max :
                        max = int(pred2[i,j])
                        maxNum = j            
            else :
                for j in range(0,3) :
                    if int(pred1[i,j]) > max :
                        max = int(pred1[i,j])
                        maxNum = j
            
            #print(pred[i], end = ' ')
            outstr = outstr + str(maxNum)
            if (i+1)%8 == 0 and i != 0 :
                print(outstr)
                output.write(outstr)
                outstr = ''
                output.write("\n")

            #print(maxNum)
            #if (i+1) % 8 == 0 : 
                #print("")
            #    output.write("\n")

        output.write("\n")        

        #print(loss1.item())
        #print(loss2.item())