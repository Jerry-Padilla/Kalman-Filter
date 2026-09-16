"""Constant-velocity Kalman filter for motion along one spatial axis.

State: [position, velocity]. Observations contain position only.
Process noise is an independent, constant acceleration each time step.
"""

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]

import numpy as np


def validate_number(name, value, *, positive=False):
    """Validate a finite, nonnegative (or strictly positive) scalar."""
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        constraint = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {constraint}")
    return value


class KalmanFilter:
    """Estimate position and velocity from noisy position observations.

    procVar is acceleration variance; sensorVar is measurement variance.
    Original constructor names and run/showGraph entry points are retained.
    """

    def __init__(
        self,
        dtime=1.0,
        velx=0.0,
        posix=0.0,
        procVar=0.2,
        sensorVar=0.5,
        initial_position_var=10.0,
        initial_velocity_var=10.0,
    ):
        self.dt = validate_number("dtime", dtime, positive=True)
        self.procVar = validate_number("procVar", procVar)
        self.sensorVar = validate_number("sensorVar", sensorVar, positive=True)
        self.x = np.array([posix, velx], dtype=float)
        if not np.all(np.isfinite(self.x)):
            raise ValueError("Initial position and velocity must be finite")
        self.P = np.diag(
            [
                validate_number("initial_position_var", initial_position_var),
                validate_number("initial_velocity_var", initial_velocity_var),
            ]
        )
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        acceleration_effect = np.array([0.5 * self.dt**2, self.dt])
        self.Q = self.procVar * np.outer(acceleration_effect, acceleration_effect)
        self.H = np.array([1.0, 0.0])
        self.currTime = 0.0
        self.times = [0.0]
        self.pos = [float(posix)]
        self.vel = [float(velx)]
        self.acc = [0.0]
        self.meas = [np.nan]  # No observation at t=0.
        self.position_variances = [float(self.P[0, 0])]

    def __str__(self):
        return (
            f"Time: {self.currTime:.2f} s | Position: {self.x[0]:.3f} | "
            f"Velocity: {self.x[1]:.3f}"
        )

    def predict(self):
        """Advance the state and covariance by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.currTime += self.dt
        return self.x.copy()

    def update(self, measurement):
        """Correct the current prediction with a scalar position measurement."""
        measurement = float(measurement)
        if not np.isfinite(measurement):
            raise ValueError("measurement must be finite")
        residual = measurement - self.H @ self.x
        innovation_variance = self.H @ self.P @ self.H + self.sensorVar
        gain = self.P @ self.H / innovation_variance
        self.x = self.x + gain * residual
        # Joseph form preserves covariance symmetry and numerical stability.
        correction = np.eye(2) - np.outer(gain, self.H)
        self.P = correction @ self.P @ correction.T + self.sensorVar * np.outer(
            gain, gain
        )
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def run(self, currx):
        """Predict, update, and record one observation at the next time step."""
        measurement = float(currx)
        if not np.isfinite(measurement):
            raise ValueError("measurement must be finite")
        previous_velocity = self.x[1]
        self.predict()
        self.update(measurement)
        self.times.append(self.currTime)
        self.meas.append(measurement)
        self.pos.append(float(self.x[0]))
        self.vel.append(float(self.x[1]))
        self.acc.append(float((self.x[1] - previous_velocity) / self.dt))
        self.position_variances.append(float(self.P[0, 0]))
        return self.x.copy()

    def showGraph(self):
        """Display recorded positions and measurements from calls to run()."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.plot(self.times, self.pos, label="Filtered position", color="#087f8c")
        ax.scatter(
            self.times,
            self.meas,
            label="Measurements",
            color="#e99b38",
            s=18,
            alpha=0.6,
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="Position (m)",
            title="Position tracking with a Kalman filter",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        plt.show()
        return fig
