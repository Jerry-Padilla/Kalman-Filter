'''
main.py: This code will give a simulation of what the Kalman Filter does as it tracks
a vehicle along its Trajectory during motion
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]


__version__ = "1.0"
__maintainer__ = "Rob Knight"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Incomplete"

#imports
from KalmanFilter import KalmanFilter
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#TODO: EVERYTHING

# Press the green button in the gutter to run the script.


def main():
    print('Welcome to Jerry\'s Kalman Filter implementation')
    K1 = KalmanFilter()
    print(K1)
    velocities_x = [0,2,3,4,5,6,7,8,8,9,11,14,18,25,34,35,33,31]

    for v in velocities_x:
        K1.run()
        print(K1)
main()





