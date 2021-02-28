from  scipy import signal
import numpy as np
import math
import matplotlib.pyplot as plt

# Reference code :https://www.dsprelated.com/freebooks/filters/Peaking_Equalizers.html
#Taken for Digital Signal Processing Udo Zolzer

def peaking_Filter(gain,fc,fs,qf):

    Q=qf 
    wcT = 2*math.pi*fc/fs;

    K=math.tan(wcT/2);
    V=gain;

    b0 =  1 + V*K/Q + K**2;
    b1 =  2*(K**2 - 1);
    b2 =  1 - V*K/Q + K**2;
    a0 =  1 + K/Q + K**2;
    a1 =  2*(K**2 - 1);
    a2 =  1 - K/Q + K**2;
    
    A = np.array([a0, a1, a2]) / a0;
    B = np.array([b0 ,b1 ,b2]) / a0;

   
    w, h= signal.freqz(B,A);
   
    plt.subplot(2, 1, 1)    
    plt.plot(fs*w/2/np.pi, np.abs(h), 'b')
    plt.plot(fc, 0.5*np.sqrt(2), 'r')
    plt.axvline(fc, color='r')
    plt.xlim(0, fs/2)
    plt.title("Lowpass Filter Frequency Response")  
    plt.xlabel('Frequency [Hz]')
    return [B,A]
  