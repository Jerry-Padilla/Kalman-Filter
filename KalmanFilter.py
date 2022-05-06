'''
KalmanFilter.py: This code will do the necessary calculations for the Kalman Filter as it is applied to
a 2 dimensional application.
'''

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]

__version__ = "1.0"
__email__ = "gerardopadillareynoso@gmail.com"
__status__ = "Complete"

from collections import namedtuple

# imports
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======

>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67
import numpy as np
import matplotlib
from scipy.linalg import solve
from filterpy.stats import plot_covariance_ellipse

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

<<<<<<< HEAD

def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


class KalmanFilter:
    # positions
    pos = []
    # velocities
    vel = []
    # accelerations
    acc = []

    def __init__(self, dtime=1, velx=0, posix=0, procVar=.2):
        self.currTime = 0
        self.procVar = procVar
        self.dt = dtime

        self.pos = [posix]
        self.meas = [posix]
        # velocities
        self.vel = [velx]

        # displacement to add to x
        self.processModel = gaussian(self.dt, self.procVar)

        self.sensorVar = .5

        self.x = gaussian(1, 10)
=======
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
>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67

    def __str__(self):
        return 'Time: {self.currTime} \n' \
               'Position: X={self.pos[0]} Y={self.pos[1]} \n' \
               'Velocity: X={self.vel[0]} Y={self.vel[1]} \n' \
               'Acceleration: X={self.acc[0]} Y={self.acc[1]}\n '.format(self=self)

    def run(self, currx):
        self.currx = currx
        self.meas.append(self.currx)
        self.predict()
        self.update()

<<<<<<< HEAD
    def predict(self):
        print('Prediction Step at time = {self.currTime}'.format(self=self))

        self.prv = gaussian(self.x.mean + self.processModel.mean, self.x.var + self.processModel.var)
=======

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
>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67

    #Covariance Update
    # P = (I− KH) P¯
    P = P - np.dot(np.dot(K, H), P))

<<<<<<< HEAD
        probs = gaussian(self.currx, self.sensorVar)
        print("probability:", probs)
        print("previous:", self.prv)
        self.x = gaussian_multiply(probs, self.prv)
        print(self.x)

        # Add our values into the Arrays
        print("xmean", self.x.mean)
        self.currx = self.x.mean
        self.pos.append(self.currx)

        self.vel.append((self.pos[len(self.pos) - 2] - self.currx) / self.dt)
        self.acc.append((self.vel[len(self.vel) - 2] - self.vel[len(self.vel) - 1]) / self.dt)
        self.currTime += self.dt

    def showGraph(self):
        plt.plot(np.array(range(len(self.pos))), self.pos, 'b', label="Position from Filter")
        plt.plot(np.array(range(len(self.pos))), self.meas, 'r^', label="measurements")
        #plt.plot(np.array(range(len(self.pos))), [1] * len(self.pos), 'g', label="Avg position")
        #plt.plot(np.array(range(len(self.vel))), self.vel, 'o--', label='Velocity')
        # plt.plot(np.array(range(len(self.acc) )), self.acc, 'g')
        plt.title("Position Vs Time")
        plt.legend(loc='best', shadow=True, fontsize='x-large')
        plt.show()
=======
    # self.currTime += self.deltaTime;
>>>>>>> 00f7637f3ded31390d3c070140d40f5deb575c67
