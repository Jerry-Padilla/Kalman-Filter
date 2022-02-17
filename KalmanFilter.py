'''
Kalman Filter.py: This code will do the necessary calculations for the Kalman Filter as it is applied to
a 2 dimensional application.
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]


__version__ = "1.0"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Incomplete"

#imports
import numpy as np


class KalmanFilter:

    def __init__(self):
        #these variables have the x and y components in a 2D plane for our application

        # time (t)
        self.currTime = 0
        # x
        self.pos =[0,0]
        # v
        self.vel =[0,0]
        # A
        self.acc =[0,0]
        # position X at time t-1 (tm1)
        self.xtm1 = 0
        #Time interval between changes.
        self.deltaTime = 0

    def __str__(self):

        return  'Time: {self.currTime} \n' \
                'Position: X={self.pos[0]} Y={self.pos[1]} \n' \
                'Velocity: X={self.vel[0]} Y={self.vel[1]} \n'\
                'Acceleration: X={self.acc[0]} Y={self.acc[1]}\n'.format(self=self)

    def run(self):
        print("Running...")
        self.predict()
        self.update()

    def predict(self):

        print('Prediction Step at time = {self.currTime}'.format(self=self))


    def update(self):
        print('Update Step at time = {self.currTime}\n'.format(self=self))

        self.currTime += 1
