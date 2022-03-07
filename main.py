'''
main.py: This code will give a simulation of what the Kalman Filter does as it tracks
a vehicle along its Trajectory during motion
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]


__version__ = "1.0"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Incomplete"

#imports
from KalmanFilter import KalmanFilter
import numpy as np
import math


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.





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




def main():
    print('Welcome to Jerry\'s Kalman Filter implementation')

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


    K1 = KalmanFilter(dt)
    #print(K1)


   # for p in position_x:
    K1.run()
    #    print(K1)


main()
