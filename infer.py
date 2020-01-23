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
    TH = 0.2

    # Training End, infer #
    input = KshDataset.music_load(filename)    
    input = input.reshape(input.shape[0], 1, -1)
    input = input.to(device, dtype=torch.float)
    
    output = []
    for i in range(0,input.shape[0], batch):
        if i+batch < input.shape[0] :
            pred = model(input[i:i+batch])
            pred = pred.to(torch.device("cpu"))
            if i == 0 :     
                softmax = nn.Softmax(dim=1)
                pred = softmax(pred)                   
                output = pred.tolist()
            else :
                pred = pred.tolist()
                for i in pred:
                    output.append(i)
    #print(output.shape)
    #torch.save(model.state_dict(), "./model/model.pth")

    index = 0
    beforeIndex = 0
    note_time_Stamp_output = []
    fx_time_Stamp_output = []

    for time in output :
        #nt(time.index(max(time)), time[time.index(max(time))])
        
        if time.index(max(time)) == 1 and time[time.index(max(time))] > 0.2 and index > beforeIndex + 1:
            note_time_Stamp_output.append(index)
            beforeIndex = index

        if time.index(max(time)) == 2 and time[time.index(max(time))] > 0.2 and index > beforeIndex + 1:
            fx_time_Stamp_output.append(index)
            beforeIndex = index
            
        if time.index(max(time)) == 3 and time[time.index(max(time))] > 0.2 and index > beforeIndex + 1:
            note_time_Stamp_output.append(index)
            fx_time_Stamp_output.append(index)
            beforeIndex = index
        
        index = index + 1

    #print(note_time_Stamp_output)
    #print(fx_time_Stamp_output)
    #print(fx_time_Stamp_output)

    song = mp.Audio(filename = (filename),  note_timestamp = note_time_Stamp_output, fx_timestamp = fx_time_Stamp_output)
    song.synthesize(diff='ka')
    song.save(filename = savename)

    return note_time_Stamp_output, fx_time_Stamp_output

def main():
    
    model = voltexNet()
    model.load_state_dict(torch.load("./train_model.pth"))
    #print ("load model")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # move model to the right device
    model.to(device)
    #input = torch.rand(128,3,80,15)

    batch = 256
    song_index = 0
    best_Acc = 0

    epoch_loss = 0.0

    infer(model, device, batch, "./Asset/albida.ogg","./test_Output/infer.wav")
    infer(model, device, batch, "./Asset/nofx.ogg","./test_Output/infer2.wav")
    infer(model, device, batch, "./Asset/KANA-BOON - Silhouette.ogg","./test_Output/infer3.wav")
    infer(model, device, batch, "./Asset/bgm.ogg","./test_Output/infer4.wav")
    


if __name__ == "__main__":
    main()
    