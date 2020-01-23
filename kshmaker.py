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
from infer import infer
import codecs

import random

def kshMaker(note_time_Stamp_output, fx_time_Stamp_output, bpm) :
    note_time_Stamp_output = [(i * 0.04)/(60/bpm/4) for i in note_time_Stamp_output]
    fx_time_Stamp_output = [(i * 0.04)/(60/bpm/4) for i in fx_time_Stamp_output]
    print(fx_time_Stamp_output)
    f = codecs.open("./test_Output/test_ksh/output.txt", 'w', 'utf-8')

    index = 0
    f.write('--\r\n')
    ksh_list = ["0000|00|--\r\n" for i in range(3200)]
    for i in note_time_Stamp_output:
        code = random.randint(1, 4)
        ksh_list[int(i)] = "1111|00|--\r\n"

    for i in fx_time_Stamp_output:
        #print(int(i))
        if ksh_list[int(i)] == "1111|00|--\r\n" :
            ksh_list[int(i)] = "1111|11|--\r\n"
        else :
            ksh_list[int(i)] = "0000|11|--\r\n"

    for i in ksh_list :
        if index%16 == 0:
            f.write('--\r\n')
        f.write(i)
        index += 1
                
    '''
    for stamp in self.note_timestamp:
        if stamp*(self.samplerate/25)+kalen < self.data.shape[0]:
            self.data[int(stamp*(self.samplerate/25)):int(stamp*(self.samplerate/25))+kalen] += kasound

    fx_cut_sw = False
    for stamp in self.fx_timestamp:
        if stamp*(self.samplerate/25)+donlen < self.data.shape[0]:                       
            self.data[int(stamp*(self.samplerate/25)):int(stamp*(self.samplerate/25))+donlen] += donsound
    '''

    f.write('--')
    print('end')

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
    
    n_arr, f_arr = infer(model, device, batch, "./Asset/bgm.ogg","./test_Output/infer.wav")
    kshMaker(n_arr, f_arr, 156)
    #n_arr, f_arr = infer(model, device, batch, "./Asset/nofx.ogg","./test_Output/infer.wav")
    #kshMaker(n_arr, f_arr, 100)
    #kshMaker(infer(model, device, batch, "./Asset/KANA-BOON - Silhouette.ogg","./test_Output/infer3.wav")_
    #kshMaker(infer(model, device, batch, "./Asset/bgm.ogg","./test_Output/infer4.wav"))


if __name__ == "__main__":
    main()