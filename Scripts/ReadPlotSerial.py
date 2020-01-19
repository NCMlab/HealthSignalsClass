# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# from numpy import convolve as np_convolve
# from scipy.signal import fftconvolve, lfilter, firwin
# from scipy.signal import convolve as sig_convolve
# from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import numpy as np
import serial 
import os
import sys
import select
#ser = serial.Serial('/dev/ttyACM0', 57600)
ser = serial.Serial('/dev/cu.usbmodem1421', 57600)
# Try this little fellow
ser.read_all()

plt.ion()

h = plt.plot()
plt.setp(h,linewidth = 1)
fs = 200 # At the Arduino
PlotLength = 10*fs # This seems to slow everything down by ovrflowing a buffer I think
tempAlldata = [0]*PlotLength
Pos = 0
FFTSize = 256
Ts = 1.0/fs
freqs = np.fft.fftfreq(FFTSize, Ts)
idx = range(2,FFTSize/2)
# Clear all data inthe Serial buffer
ser.flush()
ser.read_all()
ntaps= 7
# b = firwin(ntaps, [0.05, 0.95], width=0.05, pass_zero=False)
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.ion()
#ax2.ion()
while True:
    plt.clf()
    plt.ylim(-100,2000)
    # 10 datapoints read at a time
    NDataPoints = 10
    #data = ser.readlines(180) # This is te numbrer of 
    #NDataPoints = 100
    #data = ser.readlines(1975) # This is te numbrer of 
   
    count = 0
    data = ser.read_all()
    NDataPoints = len(data)  
    NDataPoints
    data = data.decode("utf-8").split('\n')
    NDataPoints = len(data)
    print(NDataPoints)
    if NDataPoints >= PlotLength:
        data = data[-1*PlotLength]
    count = NDataPoints
    # Take the data that already exists in the array and move it
    tempAlldata[0:(PlotLength-NDataPoints)] = tempAlldata[(NDataPoints):]
    RunSum = 0
    
    for i in data:
        if len(i) > 0:
            try:
                
                tempAlldata[-1*count] = float(i.split(',')[0])
                
#                RunSum += float(i.split(',')[0])
            except:
                print("bad data point")
                tempAlldata[-1*count] = tempAlldata[-1*(count+1)]
            count = count - 1
        else:
            tempAlldata[-1*count] = tempAlldata[-1*(count+1)]
            count = count - 1
            
            
        #print(i.split(',')[0])
    #for i in data:
    #    temp[count]=(i.decode("utf-8").strip().split(',')[0])
    #    count = count + 1
    # take the new data and append it to the Plot
#    if Pos < PlotLength:
#        tempAlldata[Pos:Pos+NDataPoints] = temp
#        Pos = Pos + NDataPoints
#    else:
#        tempAlldata = tempAlldata[NDataPoints:]+temp
    # replace zeros
#    MeanData = np.mean(tempAlldata)
#    tempAlldata = tempAlldata - MeanData
    #conv_result = sig_convolve(tempAlldata, b[np.newaxis, :], mode='valid') 
    ps = np.abs(np.fft.fft(tempAlldata[-FFTSize:]))
    h = plt.plot(freqs[idx],ps[idx])
    #h = plt.plot(tempAlldata)
    plt.setp(h,linewidth=1)
    plt.draw()



    plt.pause(0.3)

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = raw_input()
        break

