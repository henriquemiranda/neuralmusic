#
# Henrique Pereira Coutada Miranda
# 11/2015
#
from scipy.io import wavfile
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def convert2wav(filename):
    wavfilename = os.path.splitext(filename)[1]+'.wav'

    #check if it's not already a wav file and wav doesn't exist
    if wavfilename != filename and not os.path.isfile(wavfilename):
        #convert an audio file to wav using ffmpeg
        os.system("ffmpeg -i %s %s"%(filename, wavfilename))

    return wavfilename

def HSL2RGB(hsl):
    H,S,L = hsl
    #http://www.rapidtables.com/convert/color/hsl-to-rgb.htm
    #if 0 < H < 360, 0 < S < 1, 0 < L < 1
    C = (1 - abs(2*L-1)) * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = L - C / 2
    if     0 <= H <  60: R,G,B = (C, X, 0)
    elif  60 <= H < 120: R,G,B = (X, C, 0)
    elif 120 <= H < 180: R,G,B = (0, C, X)
    elif 180 <= H < 240: R,G,B = (0, X, C)
    elif 240 <= H < 300: R,G,B = (X, 0, C)
    else:                R,G,B = (C, 0, X)
    return (R+m, G+m, B+m)

def fftToPixel(fft):
    afft = abs(fft)
    afft[afft<1e-10] = 1e-10
    #H = (H+math.pi*0.5)/math.pi*360
    H = map(lambda v: math.atan2(v.imag, v.real)/math.pi+180, fft)
    L = np.log(afft)*(1./np.log(afft.max()))
    S = np.array([1.0]*len(fft)) #lame...
    return np.array(map(lambda hsl: HSL2RGB(hsl), zip(H,S,L)))

class WavToImage():
    """ Split audio samples and fft them
    """
    _split = 2**12

    def __init__(self, filename, path='.'):
        #open using scipy
        self.samplerate, self.wavfile = wavfile.read(filename)
        self.oldsamples, self.stereo = self.wavfile.shape

        #calculate duration in seconds
        self.duration = self.oldsamples/self.samplerate 

        #cut the audio (because an fft of a power fo 2 is faster)
        self.samples  = self.oldsamples / self._split * self._split #(integer division)
        self.wavfile  = self.wavfile[0:self.samples]
        self.segments = self.samples / self._split

    def get_spectrogram(self):
        """ Calculate the spectrogram
            Input: self.wavfile
            Output: self.wavfile_fft
        """
        self.wavfile_fft = np.zeros([self.stereo,self.segments,self._split/2+1],dtype=complex)
        for i in range(self.stereo):
            wavfile_fft = self.wavfile_fft[i]
            #split the samples in segments of size = self.samples/self.split
            sample_segments = self.wavfile[:, i].reshape(self.segments, self._split)
            #fft that shit
            for n in range(self.segments):
                wavfile_fft[n] = np.fft.rfft(sample_segments[n])

    def get_wavfile_from_spectrogram(self):
        """ Calculate the inverse of the spectrogram
            Input: self.wavfile_fft
            Output: wavfile
        """
        wavfile = np.zeros([self.segments,self._split,self.stereo]) 
        for i in range(self.stereo):
            sample = wavfile[:,:,i]
            wavfile_fft_segments = self.wavfile_fft[i]
            #shit that fft
            for n in range(self.segments):
                sample[n] = np.fft.irfft(wavfile_fft_segments[n])
        return wavfile.reshape([self.samples,self.stereo])

    def plot_spectrogram(self,channel=0):
        """ Plot the historgram using matplotlib
        """
        tfft = self.wavfile_fft[channel].T
        im = fftToPixel(tfft.flatten()).reshape(tfft.shape[0], tfft.shape[1], 3)
        plt.imshow(im, origin='lower')
        plt.show()

    def __str__(self):
        """ Get data about the audio
        """
        s = ""
        s += "stereo: %d\n"%self.stereo
        s += "samplerate: %d\n"%self.samplerate
        s += "duration: %ds\n"%self.duration 
        s += "old samples: %d\n"%self.oldsamples
        s += "new samples: %d\n"%self.samples 
        return s

if __name__ == "__main__":
    w = WavToImage(convert2wav("dongiovanni.mp3"))
    print w
    w.get_spectrogram()
    w.plot_spectrogram()
    wavfile = w.get_wavfile_from_spectrogram()
    print "similar array?", np.allclose(w.wavfile,wavfile)

