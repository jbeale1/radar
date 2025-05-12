import numpy as np
from typing import Tuple
from filterpy.kalman import KalmanFilter

def get_spikes(arr: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """Return indices where discrete difference is above threshold."""
    diffs = np.diff(arr)
    indices = np.where(np.abs(diffs) > threshold)[0]
    max_diff = np.max(diffs)
    return indices, max_diff

def moving_avg(arr: np.ndarray, N: int) -> np.ndarray:
    """Replace element with average of N elements."""
    if N % 2 == 0:
        raise ValueError("N should be an odd number for symmetric averaging")

    pad_width = N // 2
    padded = np.pad(arr, pad_width, mode='edge')
    kernel = np.ones(N) / N

    # Convolve with 'valid' to get the same size as input
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

def clean_spikes(speed_kmh: np.ndarray, a_max: float = 5) -> np.ndarray:
    """Remove spikes faster than a_max in m/s^2."""
    speed_ms = speed_kmh / 3.6   # convert to m/s
    dt = 0.09  # 90 ms
    acceleration = np.diff(speed_ms) / dt
    acceleration = np.insert(acceleration, 0, 0)

    valid_mask = np.abs(acceleration) <= a_max
    expanded_mask = binary_dilation(valid_mask, structure=np.ones(3))
    cleaned_speed_ms = speed_ms.copy()

    # Use linear interpolation for invalid segments
    invalid_indices = np.where(~expanded_mask)[0]
    valid_indices = np.where(expanded_mask)[0]
    cleaned_speed_ms[invalid_indices] = np.interp(invalid_indices, valid_indices, cleaned_speed_ms[valid_indices])
    cleaned_speed_kmh = cleaned_speed_ms * 3.6
    return cleaned_speed_kmh

def kalman_filter(speeds: np.ndarray, direction: int, dt: float = 0.09) -> np.ndarray:
    """Apply Kalman filtering to speed measurements."""
    if (direction < 0): # going away from sensor, better to reverse time
        speed_kmh = speeds[::-1]
    else:
        speed_kmh = speeds        
    n = len(speed_kmh)
    speed_measurements = speed_kmh / 3.6  # convert to m/s
    
    # State: [position, velocity, acceleration]
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([[0.], [speed_measurements[0]], [0.]])
    kf.F = np.array([[1, dt, 0.5*dt**2],
                    [0, 1, dt],
                    [0, 0, 1]])
    kf.H = np.array([[0, 1, 0]])  # still only measuring velocity

    kf.P *= 1000.
    kf.R = 1.0
    kf.Q = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2,   dt],
        [dt**2/2, dt,      1]
    ]) * 0.5  # Tune this

    filtered_speed = []

    for z in speed_measurements:
        kf.predict()
        kf.update(z)
        filtered_speed.append(kf.x[1, 0])  # velocity component

    if (direction < 0):
        out = filtered_speed[::-1]
    else:
        out = filtered_speed

    return np.array(out) * 3.6  # back to km/h
