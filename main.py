"""Reproducible vehicle-tracking simulation and visualization."""

__author__ = "Gerardo R Padilla Jr."
__credits__ = ["Gerardo R Padilla Jr.", "Adyasha Mohanty"]

import argparse
from pathlib import Path

import numpy as np

from KalmanFilter import KalmanFilter, validate_number


def simulate_motion(steps=100, dt=1.0, initial_position=0.0,
                    initial_velocity=5.0, process_var=0.2,
                    measurement_std=3.0, seed=42):
    """Return times, true positions, true velocities, and noisy positions.

    Arrays include t=0. Independent acceleration noise is held constant within
    each step, matching the filter's process model.
    """
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    dt = validate_number("dt", dt, positive=True)
    process_var = validate_number("process_var", process_var)
    measurement_std = validate_number("measurement_std", measurement_std, positive=True)
    if not np.all(np.isfinite([initial_position, initial_velocity])):
        raise ValueError("Initial position and velocity must be finite")
    rng = np.random.default_rng(seed)
    times = np.arange(steps + 1) * dt
    positions = np.empty(steps + 1)
    velocities = np.empty(steps + 1)
    positions[0], velocities[0] = initial_position, initial_velocity
    for i in range(1, steps + 1):
        acceleration = rng.normal(0.0, np.sqrt(process_var))
        positions[i] = positions[i - 1] + velocities[i - 1] * dt + 0.5 * acceleration * dt ** 2
        velocities[i] = velocities[i - 1] + acceleration * dt
    measurements = positions + rng.normal(0.0, measurement_std, steps + 1)
    return times, positions, velocities, measurements


def plot_results(times, truth, true_velocity, measurements, tracker):
    """Build the dashboard without displaying it, for interactive or file use."""
    import matplotlib.pyplot as plt

    estimate = np.asarray(tracker.pos)
    uncertainty = 1.96 * np.sqrt(tracker.position_variances)
    colors = {"truth": "#263445", "filter": "#087f8c", "sensor": "#d88923"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                 layout="constrained", height_ratios=[2, 1, 1])
        fig.set_facecolor("#f7f9fc")
        fig.suptitle("Kalman filter | Tracking through the noise", fontsize=19,
                     fontweight="bold", color=colors["truth"])
        axes[0].plot(times, truth, color=colors["truth"], lw=1.8, label="True position")
        axes[0].scatter(times[1:], measurements[1:], color=colors["sensor"],
                        s=17, alpha=0.55, label="Noisy measurements")
        axes[0].plot(times, estimate, color=colors["filter"], lw=2, label="Kalman estimate")
        axes[0].fill_between(times, estimate - uncertainty, estimate + uncertainty,
                             color=colors["filter"], alpha=0.15,
                             label="95% position interval")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend(loc="best", ncol=2, frameon=False, fontsize=9)
        axes[1].plot(times, true_velocity, color=colors["truth"], label="True velocity")
        axes[1].plot(times, tracker.vel, color=colors["filter"], lw=2, label="Estimated velocity")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend(loc="best", frameon=False, ncol=2, fontsize=9)
        axes[2].axhline(0, color=colors["truth"], lw=0.8)
        axes[2].plot(times[1:], measurements[1:] - truth[1:], color=colors["sensor"],
                     alpha=0.65, label="Measurement error")
        axes[2].plot(times[1:], estimate[1:] - truth[1:], color=colors["filter"],
                     lw=1.8, label="Filter error")
        axes[2].set(xlabel="Time (s)", ylabel="Position error (m)")
        axes[2].legend(loc="best", frameon=False, ncol=2, fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.15)
            ax.margins(x=0.01)
        return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps (default: 100)")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step in seconds (default: 1)")
    parser.add_argument("--initial-position", type=float, default=0.0, help="Initial position in meters")
    parser.add_argument("--initial-velocity", type=float, default=5.0, help="Initial velocity in m/s")
    parser.add_argument("--process-var", type=float, default=0.2, help="Acceleration variance in (m/s^2)^2")
    parser.add_argument("--measurement-std", type=float, default=3.0, help="Measurement standard deviation in meters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--save", type=Path, help="Save the chart, e.g. outputs/tracking.png")
    parser.add_argument("--no-show", action="store_true", help="Run without opening a chart window")
    args = parser.parse_args(argv)
    try:
        times, truth, velocity, measurements = simulate_motion(
            args.steps, args.dt, args.initial_position, args.initial_velocity,
            args.process_var, args.measurement_std, args.seed)
        tracker = KalmanFilter(dtime=args.dt, posix=args.initial_position,
                               velx=args.initial_velocity, procVar=args.process_var,
                               sensorVar=args.measurement_std ** 2)
    except ValueError as exc:
        parser.error(str(exc))
    # The initial state is the prior; the first observation is at t=dt.
    for measurement in measurements[1:]:
        tracker.run(measurement)
    measurement_rmse = np.sqrt(np.mean((measurements[1:] - truth[1:]) ** 2))
    filter_rmse = np.sqrt(np.mean((np.asarray(tracker.pos[1:]) - truth[1:]) ** 2))
    print(f"Kalman tracking | {args.steps} steps | dt={args.dt:g} s | seed={args.seed}")
    print(f"Measurement RMSE: {measurement_rmse:.3f} m")
    print(f"Filter RMSE:      {filter_rmse:.3f} m")
    if measurement_rmse > 0:
        print(f"Error reduction:  {100 * (1 - filter_rmse / measurement_rmse):.1f}%")
    if args.save or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot_results(times, truth, velocity, measurements, tracker)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, dpi=180, facecolor=fig.get_facecolor())
            print(f"Chart saved to {args.save}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    return tracker


if __name__ == "__main__":
    main()
