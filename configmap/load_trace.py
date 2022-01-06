import os
import numpy as np

def load_trace(mean, std, repeat = 10):
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for i in range(repeat):
        cooked_time = []
        cooked_bw = []

        traces = np.clip(np.random.normal(mean, std, 600), 0.1, 100.)
        for time in range(600):
            cooked_time.append(float(time / 2.))
            cooked_bw.append(float(traces[time]))

        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(str(i))

    return all_cooked_time, all_cooked_bw, all_file_names
