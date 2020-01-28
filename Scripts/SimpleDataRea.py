

import matplotlib.pyplot as plt
import numpy as np
import serial 
import os
import sys
import select
import time
from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange


#ser = serial.Serial('/dev/ttyACM0', 57600)
ser = serial.Serial('/dev/cu.usbmodem14101', 19200)

fs = 200
Ts = 1/fs
WindowSize = 600
Window = np.zeros(WindowSize)

UpdateSize = 200
Update = np.zeros(UpdateSize)


plt.plot(Window)
plt.show()

    
    
plt.clf()
while True:
    count = 0
    while count < UpdateSize:
        Update[count] = getData(ser)
        count += 1
        plt.pause(Ts)
    Window = AddData(Update, Window, UpdateSize, WindowSize)
    plt.clf()
    plt.plot(Window)
    plt.show()
    

def getData(ser):
    d = ser.readline()
    d = int(d.decode("utf-8").split('\n')[0])
    return d


def AddData(Update, Window, UpdateSize, WindowSize):
    Window[0:WindowSize-UpdateSize] = Window[UpdateSize:WindowSize]
    Window[WindowSize-UpdateSize:] = Update
    return Window