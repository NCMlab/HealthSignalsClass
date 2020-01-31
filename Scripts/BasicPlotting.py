import numpy as np
import matplotlib.pyplot as plt
import serial 

ser = serial.Serial('/dev/cu.usbmodem14101', 19200)

N = 200
BytesToCollect = N*4 - 1
fs = 200

d = np.zeros(N)
while True:
    try:
        plt.clf()
        # data = ser.readlines(BytesToCollect)
        data = ser.read_all()
        # len(data)
        # count = 0
        # for i in data:
        #     d[count] = i#i.decode("utf-8").split('\n')[0]
        #     count += 1
        d=data.decode("utf-8").split('\n')   
        d = d[0:-1]
        plt.plot(d)
        plt.draw()
    except:
        print("Error")
        pass
    plt.pause(1)
ser.close()
    

