import numpy as np

from scipy.signal import savgol_filter

def smooth_positions(prev_state, next_state, window_size=3, mode='savgol', savgol_window=5, savgol_polyorder=3):
    # Concatenate prev_state and next_state
    concatenated_state = np.concatenate((prev_state, next_state), axis=0)

    # Extract x and y coordinates from concatenated_state
    x = concatenated_state[:, 0]
    y = concatenated_state[:, 1]

    if mode == 'savgol':
        # Smooth x and y coordinates using Savitzky-Golay filter
        smoothed_x = savgol_filter(x, savgol_window, savgol_polyorder)
        smoothed_y = savgol_filter(y, savgol_window, savgol_polyorder)
    else:
        # Smooth x and y coordinates using moving average
        smoothed_x = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    # Calculate the smoothed headings using arctan2
    diff_y = np.diff(smoothed_y)
    diff_x = np.diff(smoothed_x)
    smoothed_headings = np.arctan2(diff_y, diff_x)

    smoothed_next_state = np.hstack((smoothed_x[-1],smoothed_y[-1],smoothed_headings[-1]))
    return smoothed_next_state
