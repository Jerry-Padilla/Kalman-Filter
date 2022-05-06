'''
main.py: This code will give a simulation of what the Kalman Filter does as it tracks
a vehicle along its Trajectory during motion
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]

__version__ = "1.0"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Complete"

import math

import numpy as np
import math


# imports
from KalmanFilter import KalmanFilter
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


<<<<<<< HEAD
def genRandomPos(stddev=.5, process_var=.5, initialPosition=0, initialVelocity=1, dt=1, numberOfNums=100):
    "returns track, measurements 1D ndarrays"
    x= 0
    stddev = math.sqrt(stddev)
    p_std = math.sqrt(process_var)
    zs = [initialPosition]
    times = [0]

    for _ in range(numberOfNums):
        times.append(times[len(times) - 1] + dt)
        zs.append(zs[len(zs)-1] + np.random.randn() * stddev + .5*dt)
    return np.array(zs), np.array(times)
=======



def genRandomPos( stddev=2, process_var=10, initialPosition = 0, initialVelocity=1,dt=1,numberOfNums=100):

        "returns track, measurements 1D ndarrays"
        x, vel = 0., 1.
        stddev = math.sqrt(stddev)
        p_std = math.sqrt(process_var)
        xs, zs = [], []
        for _ in range(numberOfNums):
            v = initialVelocity + (np.random.randn() * p_std)
            x += v * dt
            xs.append(x)
            zs.append(x + np.random.randn() * stddev)
        return np.array(xs), np.array(zs)


>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67


def main():
    print('Welcome to Jerry\'s Kalman Filter implementation')

<<<<<<< HEAD
    initialVelocity = 5

    # num of data points
    dataPoints = 100
    # times of each generated measurement

=======
    initialPosition = 1
    initialVelocity = 5

    #num of data points
    dataPoints = 100
    #time step
    dt = 1
    #standard Deviation
    stdev  = 15
    #Process Variance
    procVar = 10
    initialPosition = 0

    position_x = genRandomPos(stdev,procVar,initialPosition ,initialVelocity,dt, dataPoints)
   # print(position_x)
>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67

    # time step
    dt = 1

<<<<<<< HEAD
    procVar = 10.  # variance in the  movement (standard Deviation Squared
=======
    K1 = KalmanFilter(dt)
    #print(K1)
>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67

    initialPosition = 0

<<<<<<< HEAD
    position_x, times = genRandomPos(math.sqrt(procVar), procVar, initialPosition, initialVelocity, dt, dataPoints)
    # print(position_x)
    '''
    plt.plot(times,position_x)
    plt.show()
    '''
    K1 = KalmanFilter(dt,procVar)
    # print(K1)

    for i in range(dataPoints):
        #print(i)
        K1.run(position_x[i])
        #print(K1)

    K1.showGraph()
=======
   # for p in position_x:
    K1.run()
    #    print(K1)

>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67

main()
