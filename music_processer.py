import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from librosa.effects import pitch_shift
import pickle
import sys
from numba import jit, prange
from sklearn.preprocessing import normalize


class Audio:
    """
    audio class which holds music data and timestamp for notes.
    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.
    Variables:
    Example:
        >>>from music_processor import *
        >>>song = Audio(filename)
        >>># to get audio data
        >>>song.data
        >>># to import .tja files:
        >>>song.import_tja(filename)
        >>># to get data converted
        >>>song.data = (song.data[:,0]+song.data[:,1])/2
        >>>fft_and_melscale(song, include_zero_cross=False)
    """

    def __init__(self, filename,note_timestamp,fx_timestamp, stereo=False):

        self.data, self.samplerate = sf.read(filename, always_2d=False)
        if stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        #print(self.data.shape)
        self.note_timestamp = note_timestamp
        self.fx_timestamp = fx_timestamp
        #self.nove_timestamp = timestamp


    def plotaudio(self, start_t, stop_t):

        plt.plot(np.linspace(start_t, stop_t, stop_t-start_t), self.data[start_t:stop_t, 0])
        plt.show()


    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)

    def synthesize(self, diff=True, ka="./data/clap.wav"):
        
        kasound = sf.read(ka)[0]
        kasound = (kasound[:, 0] + kasound[:, 1]) / 2
        kalen = len(kasound)
        
        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0]*self.samplerate)
                try:
                    if stamp[1] in (2, 4):
                        self.data[timing:timing+kalen] += kasound
                except ValueError:
                    pass
        
        elif diff == 'ka':
            #print(self.note_timestamp)
            #print(self.fx_timestamp)
            #print(self.data.shape)
            if isinstance(self.note_timestamp[0], tuple):
                for stamp in self.note_timestamp:
                    if stamp*(self.samplerate/25)+kalen < self.data.shape[0]:
                        self.data[int(stamp[0]*(self.samplerate/25)):int(stamp[0]*(self.samplerate/25))+kalen] += kasound
            else:
                for stamp in self.note_timestamp:
                    if stamp*(self.samplerate/25)+kalen < self.data.shape[0]:
                        self.data[int(stamp*(self.samplerate/25)):int(stamp*(self.samplerate/25))+kalen] += kasound

                #fx_cut_sw = False
                #for stamp in self.fx_timestamp:
                    #if stamp*(self.samplerate/25)+kalen < self.data.shape[0]:                       
                        #self.data[int(stamp*(self.samplerate/25))] -= 10

        
