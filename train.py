#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys

from torch.autograd import Variable
from dataset import KshDataset
from torch.utils.data import DataLoader

from net.model import voltexNet
import music_processer as mp
import torch.nn.functional as F
from torch.optim import lr_scheduler

from tqdm import tqdm

import os

def infer(model, device, batch, filename, savename) :

    # Training End, infer #
    input = KshDataset.music_load(filename + '/nofx.ogg')    
    input = input.reshape(input.shape[0], 1, -1)
    input = input.to(device, dtype=torch.float)
    
    output = []
    for i in range(0,input.shape[0], batch):
        if i+batch < input.shape[0] :
            pred = model(input[i:i+batch],batch)
            pred = pred.to(torch.device("cpu"))
            if i == 0 :                        
                output = pred.tolist()
            else :
                pred = pred.tolist()
                for i in pred:
                    output.append(i)
    #print(output.shape)
    #torch.save(model.state_dict(), "./model/model.pth")

    index = 0
    note_time_Stamp_output = []
    fx_time_Stamp_output = []

    for time in output :
        #print(time.index(max(time)))
        
        if time.index(max(time)) == 1 :
            note_time_Stamp_output.append(index)

        if time.index(max(time)) == 2 :
            fx_time_Stamp_output.append(index)
            
        if time.index(max(time)) == 3 :
            note_time_Stamp_output.append(index)
            fx_time_Stamp_output.append(index)
        
        index = index + 1

    print(note_time_Stamp_output)
    print(fx_time_Stamp_output)
    #print(fx_time_Stamp_output)

    song = mp.Audio(filename = (filename + "nofx.ogg"),  note_timestamp = note_time_Stamp_output, fx_timestamp = fx_time_Stamp_output)
    song.synthesize(diff='ka')
    song.save(filename = savename)

def main():
    
    model = voltexNet()
    #model.load_state_dict(torch.load("./model_99_.pth"))
    #print ("load model")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # move model to the right device
    model.to(device)
    #input = torch.rand(128,3,80,15)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    # kshDataset에서는 (곡의 길이 * 25 = batch) * (sample rate * 0.04)의 tensor가 넘어옴
    # for문은 (batch)만큼 반복
    # CNN input Tensor = (sample rate * 0.04) 1D Tensor
    # CNN output = (1,1) sigmoid unit = RNN input
    # 구현 예정
    # RNN output = (batch,1) 1D Tensor 
    # RNN Target = .ksh file => 10ms True/False TimeStamp 

    dirname = "./data/songs/"
    valname = "./data_test/songs/"
    filenames = os.listdir(dirname)
    valnames = os.listdir(valname)

    batch = 128
    song_index = 0
    best_Acc = 0

    epoch_loss = 0.0
    
    for epoch in range(0,800) :
        epoch_loss = 0.0
        train_loss = 0.0
        index = 0
        song_index = 0
        print("train...")
        for filename in tqdm(filenames):
            song_index += 1
            full_filename = os.path.join(dirname, filename)
            #print(full_filename)

            input = KshDataset.music_load("./cache/" + filename+'.npy')
            #print(input.shape)
            #input = input.reshape(input.shape[0], 1, -1)       
            target = KshDataset.timeStamp(full_filename + '/exh.ksh', input.shape[0]) 
            try : 
                if target == None:
                    continue
            except TypeError: 
                __ = 1

            input = input.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.int64)
            
            #print(input.shape)    
            #print(target.shape)
            model.train(True)  # Set model to training mode
        
            for i in range(0,input.shape[0], batch):
                if i+batch < input.shape[0] :
                    optimizer.zero_grad()

                    pred = model(input[i:i+batch])
                    
                    
                    loss = criterion(pred.squeeze(), target[i:i+batch].squeeze())
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    index += 1     
                
                else :
                    optimizer.zero_grad()
                    tmp_batch = input.shape[0] - i
                    if tmp_batch > 1 : 
                        pred = model(input[i:i+tmp_batch-1])
                        
                        loss = criterion(pred.squeeze(), target[i:i+tmp_batch-1].squeeze())
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        index += 1    
                

        scheduler.step()
        
        print("eval...")
        noneScore = 0
        acc = 0
        #model.to(torch.device("cpu"))
        for filename in tqdm(valnames):
            full_filename = os.path.join(valname, filename)
            input = KshDataset.music_load("./cache/" + filename+'.npy')
            input = input.to(device, dtype=torch.float)   
            target = KshDataset.timeStamp(full_filename + '/exh.ksh', input.shape[0]) 
        
            output = []
            for i in range(0,input.shape[0], batch):
                if i+batch < input.shape[0] :
                    pred = model(input[i:i+batch])
                    pred = pred.to(torch.device("cpu"))
                    if i == 0 :                        
                        output = pred.tolist()
                    else :
                        pred = pred.tolist()
                        for i in pred:
                            output.append(i)
                    
            
            #output = output.to(torch.device("cpu"))
            #output = output.tolist()

            #loss = criterion(pred.squeeze(), target[i*batch:i*batch+batch].squeeze())
            
            acc_tmp = 0
            try :
                for (out, tar) in zip(output, target):
                    if out.index(max(out)) == tar:
                        acc_tmp += 1

                acc += acc_tmp / input.shape[0]
            except TypeError:
                noneScore += 1
                continue
        
        if acc != 0 :
            acc = acc / (len(valnames) - noneScore)
                
        print("epoch : " + str(epoch) + "\ttrain_loss : " + str(epoch_loss/index) + "\ttest_acc : " + str(acc) + "\n")
        
        if acc > best_Acc :
            best_Acc = acc
            torch.save(model.state_dict(), "./model/model_bestAcc_.pth")
            print("save " + str(best_Acc))

        if best_Acc > 0.9 :
            break
        
        torch.save(model.state_dict(), "./model/model_"+ str(epoch)+ "_.pth")
    
    


if __name__ == "__main__":
    main()
    