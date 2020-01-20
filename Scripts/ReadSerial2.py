# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

import time

t_end = time.monotonic() + 60 * 15 << the plus is how much time to wait
while time.monotonic() < t_end:
    I am thinking I can check the time at the beginning, collect data, plot, then have 
    a waiting while loop. As long as I am allowing enough time for everything to 
    get done before I restart the main loop, this should work.
Get a,b coeff
sig.butter(5,0.4) this equals a low pass cutoff of 0.2, which is 40Hz for fs=200.
>>>Get pyfda for filter design
from scipy import signal
fs =200
fcl = 10
fch = 50
b = signal.firwin(30, [2*fcl/fs, 2*fch/fs], window=('kaiser', 8),pass_zero=False)
w, h = signal.freqz(b)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)
plt.plot(fs*w/(2*np.pi), abs(h), 'b')
plt.show()


plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
"""
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import serial 
import os
import sys
import select
import time

#ser = serial.Serial('/dev/ttyACM0', 57600)
ser = serial.Serial('/dev/cu.usbmodem14101', 57600)
# Try this little fellow
# ser.readall()

fs = 200 # At the Arduino
PlotLength =  5*fs # This seems to slow everything down by ovrflowing a buffer I think
tempAlldata = [0]*PlotLength
RectData = [0]*PlotLength

Pos = 0
FFTSize = 256
Ts = 1.0/fs
freqs = np.fft.fftfreq(FFTSize, Ts)
idx = range(1,int(FFTSize/2))
# Clear all data inthe Serial buffer
ser.flush()
ser.readall()
# Set up the plotting figure
fig = plt.figure()
gs = gridspec.GridSpec(4, 2)
ax1 = fig.add_subplot(gs[0,0]) # This will be for the filter
ax2 = fig.add_subplot(gs[0,1]) # This will be for the power spectrum
ax3 = fig.add_subplot(gs[1,:]) # This will be for the unfiltered data
ax4 = fig.add_subplot(gs[2:,:]) # This will be for the filtered data


timeLimit = 0.5 # in seconds
# Create a filter here
fcl = 2.0
fch = 20.0
#print(fs)
NTaps = 151
b = signal.firwin(NTaps, [2*fcl/fs, 2*fch/fs], pass_zero=False)
bHigh = signal.firwin(NTaps, [2*4.0/fs], pass_zero=False)
#b = [0.2, 0.2, 0.2, 0.2, 0.2]

w, h = signal.freqz(b)
wHigh, hHigh = signal.freqz(bHigh)

# Plot the filter
ax1.plot(fs*w/(2*np.pi), abs(h), 'g')
#ax1.plot(fs*wHigh/(2*np.pi), abs(hHigh), 'b')
#ax1.show()
def PeakDetect(data, Threshold):
    PeakZeros = np.zeros(len(data))
    # Find values above threshold
    SupraThreshold = data > Threshold
    SupraThreshold = np.where(SupraThreshold)
    # Add a zero at the beginning
    SupraThresholdPeaks = np.insert(SupraThreshold,0,0)
    SupraThresholdPeaks = np.diff(SupraThresholdPeaks)
    SupraThresholdLocs = SupraThreshold[0][np.where(SupraThresholdPeaks>1)[0]]
    PeakZeros[SupraThresholdLocs] = Threshold
    return PeakZeros

def HeartRate(WhereAreZeros, fs):
    LastHalf = WhereAreZeros[len(WhereAreZeros)/2:]
    MeanDiff = np.mean(np.diff(np.where(LastHalf)[0]))
    HR = 60*(1/(MeanDiff/200))
    return HR
def RRintervals(WhereAreZeros,fs):
    # Take the time of each R wave and find the time between them. This will be the RRi plot
#    https://github.com/rhenanbartels/hrv/blob/develop/hrv/classical.py
    return
    
def AvgRectified(data,count):
    RectData = np.abs(data[-count:])
    AvgRectData = np.average(RectData)
    return AvgRectData
    
while True:
    
    t_end = time.time()
    ax2.cla()
    ax3.cla()
    ax4.cla()
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
    #print(NDataPoints)
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
                if float(i.split(',')[0]) < 20: # something is wrong
                    #print("bad data point")
                    tempAlldata[-1*count] = tempAlldata[-1*(count+1)]
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
    
    # Take the last chunk of data and calculate the FFT for it
    ps = np.abs(np.fft.fft(tempAlldata[-FFTSize:]))
    # Take the data and filter it
    fData1 = signal.lfilter(bHigh, 1, tempAlldata)
    fData2 = signal.lfilter(b, 1, tempAlldata)
    fData2 = fData2[NTaps+1:]
    ps1 = np.abs(np.fft.fft(fData1[-FFTSize:]))
    ps2 = np.abs(np.fft.fft(fData2[-FFTSize:]))
    # Create a rectified time series
#    RectValue = AvgRectified(fData2,NDataPoints)
#    RectData = RectData[NDataPoints:]+[RectValue]*NDataPoints

    
    # Plot the filtered data






    ax2.plot(freqs[idx],ps1[idx],'b')
    ax2.plot(freqs[idx],ps2[idx],'g')
#    ax3.plot(tempAlldata)
#    ax3.plot(fData1,'b')
    #Peaks = PeakDetect(fData2, 200)
    #HR = HeartRate(Peaks, fs)
    #print(HR)
    ax2.set_ylim([0,5000])
    ax2.set_xlabel('Frequency in Hertz')
#    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power')
#    ax2.set_ylim([0,1000])
    ax3.plot(tempAlldata)
    
    ax3.set_ylabel('Raw Data')
    ax4.plot(fData2,'g')
    #ax4.plot(Peaks,'r')
#    ax4.plot(RectData,'b')
#    ax4.set_ylabel(str(HR))
#    ax4.legend(('Filtered Data','Rectified in Chuncks'))
    #plt.setp(h,linewidth=1)
    #plt.draw()
    #plt.show()
    ProcessTime = time.time() - t_end
    #print(ProcessTime)
    while time.time() - t_end < timeLimit:
        plt.pause(0.01) # << wait 
    time.time()

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = raw_input()
        break

