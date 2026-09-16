"""Numerical regression checks; run with python -m unittest discover -s tests."""

import unittest

import numpy as np

from KalmanFilter import KalmanFilter
from main import simulate_motion


class KalmanFilterTests(unittest.TestCase):
    def test_prediction_and_scalar_update(self):
        tracker = KalmanFilter(dtime=1, posix=0, velx=1, procVar=0,
                               sensorVar=1, initial_position_var=1,
                               initial_velocity_var=1)
        np.testing.assert_allclose(tracker.predict(), [1, 1])
        np.testing.assert_allclose(tracker.P, [[2, 1], [1, 1]])
        np.testing.assert_allclose(tracker.update(2), [5 / 3, 4 / 3])
        np.testing.assert_allclose(tracker.P, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])

    def test_time_and_direction(self):
        tracker = KalmanFilter(dtime=0.25, velx=4, procVar=0)
        for position in range(1, 11):
            tracker.run(position)
        np.testing.assert_allclose(tracker.pos, np.arange(11))
        np.testing.assert_allclose(tracker.vel, 4)
        self.assertEqual(tracker.currTime, 2.5)
        self.assertEqual(len(tracker.times), len(tracker.meas))

    def test_seeded_simulation_improves_position_error(self):
        times, truth, _, measurements = simulate_motion(steps=300, dt=0.5)
        tracker = KalmanFilter(dtime=0.5, velx=5, procVar=0.2, sensorVar=9)
        for measurement in measurements[1:]:
            tracker.run(measurement)
            np.testing.assert_allclose(tracker.P, tracker.P.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(tracker.P).min(), -1e-12)
        np.testing.assert_allclose(tracker.times, times)
        filtered_mse = np.mean((tracker.pos[1:] - truth[1:]) ** 2)
        measured_mse = np.mean((measurements[1:] - truth[1:]) ** 2)
        self.assertLess(filtered_mse, measured_mse * 0.6)

    def test_simulation_respects_initial_state_and_timestep(self):
        times, positions, velocities, measurements = simulate_motion(
            steps=4, dt=0.5, initial_position=10, initial_velocity=-2, process_var=0)
        np.testing.assert_allclose(times, [0, 0.5, 1, 1.5, 2])
        np.testing.assert_allclose(positions, [10, 9, 8, 7, 6])
        np.testing.assert_allclose(velocities, -2)
        repeated = simulate_motion(steps=4, dt=0.5, initial_position=10,
                                   initial_velocity=-2, process_var=0)
        np.testing.assert_array_equal(measurements, repeated[3])

    def test_invalid_parameters(self):
        for kwargs in ({"dtime": 0}, {"procVar": -1}, {"sensorVar": 0},
                       {"velx": np.nan}, {"initial_position_var": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KalmanFilter(**kwargs)
        for kwargs in ({"steps": 0}, {"dt": -1}, {"measurement_std": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_motion(**kwargs)

    def test_invalid_measurement_does_not_advance_state(self):
        tracker = KalmanFilter()
        initial_state, initial_covariance = tracker.x.copy(), tracker.P.copy()
        with self.assertRaises(ValueError):
            tracker.run(np.nan)
        np.testing.assert_array_equal(tracker.x, initial_state)
        np.testing.assert_array_equal(tracker.P, initial_covariance)
        self.assertEqual(tracker.currTime, 0)

    def test_instances_have_independent_histories(self):
        first, second = KalmanFilter(), KalmanFilter()
        first.run(2)
        self.assertEqual(second.pos, [0])
        self.assertEqual(second.acc, [0])


if __name__ == "__main__":
    unittest.main()
