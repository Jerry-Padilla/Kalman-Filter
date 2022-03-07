'''
KalmanFilter.py: This code will do the necessary calculations for the Kalman Filter as it is applied to
a 2 dimensional application.
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]

__version__ = "1.0"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Incomplete"

# imports

import numpy as np
import matplotlib
from scipy.linalg import solve
from filterpy.stats import plot_covariance_ellipse


class KalmanFilter:
    currTime = 0
    pos = [0, 0]
    vel = [0, 0]
    acc = [0, 0]

    xtm1 = 0
    ytm1 = 0

    # time step
    deltaTime = 1

    # P (covariance) = [sigma^2 pos,0].[0,sigma^2 vel]
    P = 0
    # u (mu) =[position],[velocity ] in x component
    u = 0
    # State transition matrix, Φ Phi
    F = 0
    Q = 0

    def __init__(self, dt, velx=0, vely=0, posix=0, posiy=0):
        self.currTime = 1
        self.pos = [posix, posiy]
        self.vel = [velx, vely]
        self.acc = [0, 0]
        self.xtm1 = 0
        self.ytm1 = 0
        self.deltaTime = dt

        self.u = np.array([[self.pos[0], self.vel[0]]]).T

        self.P = np.diag([20, 900])

        self.F = np.array([[1, self.deltaTime], [0, 1]])

        # Process Noise (Q)
        self.Q = 10

    def __str__(self):
        return 'Time: {self.currTime} \n' \
               'Position: X={self.pos[0]} Y={self.pos[1]} \n' \
               'Velocity: X={self.vel[0]} Y={self.vel[1]} \n' \
               'Acceleration: X={self.acc[0]} Y={self.acc[1]}\n '.format(self=self)

    def run(self):
        print("Running...")
        self.predict()
        self.update()


def predict(self):
    print('Prediction Step at time = {self.currTime}'.format(self=self))

    # x = Fx
    self.pos[0] = np.dot(self.F, self.pos[0])

    # P = FPF' + Q
    self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    plot_covariance_ellipse(self.u, self.P, edgecolor='r', axis_equal=True,
                            title='Covariance at time = {self.currTime}'.format(self=self))
    matplotlib.pyplot.xlabel('position')
    matplotlib.pyplot.ylabel('velocity')
    matplotlib.pyplot.show()


def update(self):
    print('Update Step at time = {self.currTime}\n'.format(self=self))

    # System Uncertainty
    # S = H (dot) P¯ (dot)S (dot) H.T + R
    S = np.dot(H, np.dot(P, H.T)) + R

    #Kalman Gain
    # K = P¯ (dot) H.T (dot) S^−1
    K = np.dot(np.dot(P, H.T), inv(S)))

    #Residual
    #y = z − H (dot) x¯
    y = z - np.dot(H, x)

    # State Update
    # x = x¯ + K (dot)  y
    x += np.dot(K, y)

    #Covariance Update
    # P = (I− KH) P¯
    P = P - np.dot(np.dot(K, H), P))

    # self.currTime += self.deltaTime;
