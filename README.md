# Kalman Filter: vehicle tracking

A Python simulation showing how a Kalman filter estimates position and velocity
from noisy position measurements. The chart compares true motion, sensor readings,
and the filtered estimate, with a 95% position uncertainty interval, velocity
estimates, and position errors.

This is **one-dimensional motion with a two-element state** (position and
velocity), not a vehicle moving in an x/y plane.

## Run it

Use Python 3.10 or later. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python`.
If dependencies are already installed, `python main.py` is enough.

Save a chart without opening a window:

```powershell
python main.py --no-show --save outputs/tracking.png
```

Try different sensor noise and time steps:

```powershell
python main.py --steps 200 --dt 0.5 --measurement-std 6 --process-var 0.2 --seed 7
python main.py --help
```

The default random seed is 42, so repeated runs are reproducible. The terminal
reports measurement and filter root-mean-square error (RMSE) in meters; lower is
better. Improvement depends on noise levels and the sample, and is not guaranteed
for every run.

## How it works

1. Simulate a vehicle with an initial position and velocity. Draw a random
   acceleration each time step, and add independent sensor noise to its position.
2. Predict the next position and velocity using a constant-velocity model.
3. Correct the prediction using the noisy position measurement and its variance.

The transition matrix is `F = [[1, dt], [0, 1]]`; the observation matrix is
`H = [1, 0]`. For acceleration variance `q`, process covariance is
`Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]`. Measurement variance is
`R = measurement_std^2`. The covariance update uses the Joseph form for numerical
stability. The shaded interval is estimated position ±1.96 standard deviations
under the filter's Gaussian assumptions.

`--measurement-std` is a **standard deviation** in meters. `--process-var` is an
**acceleration variance** in `(m/s²)²`. Increasing measurement noise makes the
filter trust the model more; increasing process noise lets it follow changes in
motion more readily. The simulation uses the same noise model as the filter.

Initial position and velocity are supplied as the prior, with default variances
of 10 m² and 10 (m/s)². Updates and RMSE calculations start at `t=dt`; the initial
state is not counted as an observation.

## Use the filter directly

```python
from KalmanFilter import KalmanFilter

tracker = KalmanFilter(dtime=1, posix=0, velx=5, procVar=0.2, sensorVar=9)
for measurement in [4.8, 10.5, 14.2, 21.0]:
    position, velocity = tracker.run(measurement)
    print(tracker)
tracker.showGraph()
```

`run()` predicts, corrects, and records one step. For manual control, call
`predict()` and `update(measurement)` separately; these methods do not append to
the plotting history. Importing either module does not start the simulation.

## Verify

```powershell
python -m unittest discover -s tests -v
```

Tests cover a hand-calculated update, time alignment, covariance stability,
reproducibility, invalid inputs, independent instances, and error reduction on a
fixed simulation.

Original project by Gerardo R Padilla Jr., with credit to Adyasha Mohanty.
