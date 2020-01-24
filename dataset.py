
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from math import ceil

import os
import codecs
import numpy as np
import torch
from PIL import Image

import librosa
import librosa.display
from tqdm import tqdm
from scipy.fftpack import fft
from scipy import signal

import music_processer as mp

class KshDataset():
    def __init__(self, root):
        self.root = root     
        #print(self.imgs)
        self.len = len(self.oggs)

    # for test
    def dummyTarget(self):
        target = torch.randint(1, size=(512,))
        for i in range(0,512):
            target[i] = 0

        return target, target

    def targetToTensor(self, target):   
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
                elif target[i] == 'o':
                    output2[i] = 1
                elif target[i] == ':':
                    output2[i] = 2
                elif target[i] == '0':
                    output2[i] = 3
                elif target[i] == 'D':
                    output2[i] = 4
                elif target[i] == 'c':
                    output2[i] = 5
                elif target[i] == 'C':
                    output2[i] = 6
                else:
                    output2[i] = 7
            else :
                output1[i] = int(target[i])
        '''print("--")
        for i in range(0,len(output1)):
            print(output1[i].item(), end = ' ')
            if (i+1)%8 == 0: 
                print("")
        print("--")

        for i in range(0,len(output1)):
            print(output2[i].item(), end = ' ')
            if (i+1)%8 == 0: 
                print("")
        
        print("--")
        '''
        return output1, output2

    

    def music_load(filename) :
        '''
        y, sr = librosa.load(filename, sr=44100)
        y_  = np.zeros(int(44100*0.04)-int(len(y)%int(44100*0.04)))
        y = np.hstack([y,y_])

        y = np.reshape(y, (-1, int(sr/100)*4 ))

        y = torch.from_numpy(y)

        print(a.shape)
        '''
        y = np.load(filename)
        y = torch.from_numpy(y)

        return y

    def music_cache_make(filelist) :
        #f = codecs.open("music_cache_data","w")

        for filename in tqdm(filelist):
            y, sr = librosa.load("./data_test/songs/" + filename+"/nofx.ogg", sr=44100)
            y_  = np.zeros(int(44100*0.04)-int(len(y)%int(44100*0.04)))
            y = np.hstack([y,y_])

            y = np.reshape(y, (-1, int(sr/100)*4 ))
            #print(y.shape)

            a = []
            for i in range(0, y.shape[0]) :
                y1 = np.abs(librosa.stft(y[i], n_fft = 1764, hop_length=2048, win_length = 441))
                y2 = np.abs(librosa.stft(y[i], n_fft = 1764, hop_length=2048, win_length = 882))
                y3 = np.abs(librosa.stft(y[i], n_fft = 1764, hop_length=2048, win_length = 1764))

                a.append(np.array([y1,y2,y3]).tolist())
                #if i == 200 : print(np.array([y1,y2,y3]))

            a = np.array(a)
            #print(a.shape)
            np.save("./cache/"+filename,a)

        return y

    def timeStamp(filename, term) :
        sr = 44100

        f = codecs.open(filename, 'r+', 'utf-8')
        count_sw = False
        read_sw = False
        Beat = "4/4"
        bpm = 0

        all_beat = 0
        note_list = []
        tmp_note_list = []

        time_count = 0.0

        note_time_Stamp = [0 for i in range(term)]
        note_time_Stamp_output = []

        fx_time_Stamp = [0 for i in range(term)]
        fx_time_Stamp_output = []

        while True:
            line = f.readline()
            if not line: break  

            if line.find("beat=") != -1:
                if line != "beat=4/4\r\n" :
                    #print("error!")
                    return 

                beat = line.replace("beat=","")              
                count_sw = True
                continue          

            if line[0] == 't' and line[1] == '=' and count_sw:
                try :
                    bpm = int(line.replace("t=",""))
                except ValueError :
                    return

                read_sw = True
                continue

            if read_sw :
                if line[0] == '-' and line[1] == '-' :
                    #print(len(tmp_note_list))
                    
                    for note in tmp_note_list : 
                        note_list.append([note.replace("\r\n",""),time_count])
                        time_count = time_count + 60/bpm/(len(tmp_note_list)/4)

                    tmp_note_list = []
                    continue

                if line.find("|") == 4 :
                    tmp_note_list.append(line)
                    all_beat = all_beat + 1
                #print(line.replace("\r\n",""))
        
        tmp_note = []
        for note in note_list : 
            if note[0][0:4] != "0000" :
                note_time_Stamp[int(note[1]*(sr/(sr*0.04)))] = 1
            if note[0][5:7] != "00":
                #print(note[0][5:7],int(note[1]*100))
                for i in range(0,25) :
                    if int(note[1]*(sr/(sr*0.04)))+i < len(fx_time_Stamp) :
                        fx_time_Stamp[int(note[1]*(sr/(sr*0.04)))+i] = 1
                #print(note[0][0:4])

            tmp_note = note
            #else :

        index = 0
        for time in note_time_Stamp :
            if time == 1 :
                note_time_Stamp_output.append(index)

            index = index + 1
        #print(note_time_Stamp_output)


        index = 0
        for time in fx_time_Stamp :
            if time == 1 :
                fx_time_Stamp_output.append(index)

            index = index + 1

        # test with sound output
        #print(len(note_time_Stamp_output))
        #song = mp.Audio(filename = "./data/songs/rootsphere_lastnote/nofx.ogg",  note_timestamp = note_time_Stamp_output, fx_timestamp = fx_time_Stamp_output)
        #song.synthesize(diff='ka')
        #song.save(filename = "test.wav")
        
        note_time_Stamp = np.asarray(note_time_Stamp)
        fx_time_Stamp = np.asarray(fx_time_Stamp)

        dummy = np.zeros((int(44100*0.04)-len(note_time_Stamp)%int(44100*0.04)))
        note_time_Stamp = np.hstack([note_time_Stamp,dummy])
        fx_time_Stamp = np.hstack([fx_time_Stamp,dummy])

        #return_timestamp = np.column_stack([note_time_Stamp,fx_time_Stamp])

        
        # class Number target
        return_timestamp = []
        for (note, fx) in zip(note_time_Stamp, fx_time_Stamp) :
            if note == 1 and fx == 0 :
                return_timestamp.append([1])
            elif note == 0 and fx == 1 :
                return_timestamp.append([2])
            elif note == 1 and fx == 1 :
                return_timestamp.append([3])
            else :
                return_timestamp.append([0])
        
        # test log
        #for time in return_timestamp :
        #    print(time)
        

        return_timestamp = np.asarray(return_timestamp)
        return_timestamp = torch.from_numpy(return_timestamp)
        
        #note_time_Stamp = torch.from_numpy(note_time_Stamp)

        return return_timestamp

if __name__ == "__main__":
    filenames = os.listdir("./data_test/songs/")
    KshDataset.music_cache_make(filenames)
    #y, sr = librosa.load("./data/songs/rootsphere_lastnote/nofx.ogg", sr=44100)
    #KshDataset.timeStamp("./data/songs/rootsphere_lastnote/exh.ksh", y.shape[0])
    #print(y.shape[0]//441)