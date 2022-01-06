import numpy as np
import fixed_env as env
import load_trace
import configmap_mpc
import online_changepoint_detection as oncd
from functools import partial
import mpc

S_INFO = 5
S_LEN = 8

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent

SUMMARY_DIR = './test_results'
LOG_FILE = SUMMARY_DIR + '/log_sim_oboempc'

def trimPlayerVisibleBW(player_visible_bw, thresh):
    ret = []
    cutoff = 0
    lenarray = len(player_visible_bw)
    if lenarray <= thresh:
        return player_visible_bw, cutoff

    cutoff = lenarray - thresh
    ret = player_visible_bw[cutoff:]
    return ret, cutoff


def onlineCD(chunk_when_last_chd, player_visible_bw, interval=5):
    chd_detected = False
    chd_index = chunk_when_last_chd
    trimThresh = 1000.
    player_visible_bw, cutoff = trimPlayerVisibleBW(
        player_visible_bw, trimThresh)
    R, maxes = oncd.online_changepoint_detection(np.asanyarray(player_visible_bw), partial(
        oncd.constant_hazard, 250), oncd.StudentT(0.1, 0.01, 1, 0))
    interval = min(interval, len(player_visible_bw))
    changeArray = R[interval, interval:-1]
    # reversed(list(enumerate(changeArray))): # enumerate(changeArray):
    for i, v in reversed(list(enumerate(changeArray))):
        if v > 0.01 and i + cutoff > chunk_when_last_chd and not (i == 0 and chunk_when_last_chd > -1):
            chd_index = i + cutoff
            chd_detected = True
            break
    return chd_detected, chd_index


def getDynamicconfig_mpc(pv_list_hyb, bw, std, step=900):
    bw_step = step
    std_step = step
    bw_cut = int(float(bw)/bw_step)*bw_step
    std_cut = int(float(std)/std_step)*std_step
    current_list_hyb = list()
    count = 0
    if True:
        if bw == -1 and std == -1:
            return 0.0, 0.0, 0.0
        # if key not in performance vector
        if (bw_cut, std_cut) not in list(pv_list_hyb.keys()):
            for i in range(2, 1000, 1):
                count += 1
                for bw_ in [bw_cut - (i - 1) * bw_step, bw_cut + (i-1) * bw_step]:
                    for std_ in range(std_cut - (i - 1) * std_step, std_cut + (i-1) * std_step + std_step, std_step):
                        if (bw_, std_) in list(pv_list_hyb.keys()):
                            current_list_hyb = current_list_hyb + \
                                pv_list_hyb[(bw_, std_)]
                for std_ in [std_cut - (i - 1) * std_step, std_cut + (i-1) * std_step]:
                    for bw_ in range(bw_cut - (i - 2) * bw_step, bw_cut + (i-1) * bw_step, bw_step):
                        if (bw_, std_) in list(pv_list_hyb.keys()):
                            current_list_hyb = current_list_hyb + \
                                pv_list_hyb[(bw_, std_)]
                if len(current_list_hyb) == 0:
                    continue
                else:  # len(abr_list)>0 and 'BB' not in abr_list:
                    break
        else:
            current_list_hyb = current_list_hyb + \
                pv_list_hyb[(bw_cut, std_cut)]

    if len(current_list_hyb) == 0:
        return 0.0, 0.0, 0.0
    if max(current_list_hyb) == -1.0:
        return 0.0, 0.0, 0.0
    return min(current_list_hyb), np.percentile(current_list_hyb, 50), max(current_list_hyb)

def main():
    np.random.seed()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        './oboe_traces/')

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    ccmpc = mpc.mpc()
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    throughput = []

    video_count = 0
    ch_index = -1
    discount = 0.5
    state = np.zeros((S_INFO, S_LEN))

    while True:
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(discount) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        state = np.roll(state, -1, axis=1)
        
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = float(delay) / M_IN_K

        # ================== MPC =========================
        throughput.append(state[3, -1])
        bandwidth = np.array(throughput) * 8000.

        ch_detected, ch_idx = onlineCD(ch_index, bandwidth)
        if ch_detected:
            ch_index = ch_idx
            bandwidth_arr = bandwidth[ch_index:]
            mean_bandwidth = np.mean(bandwidth_arr)
            std_bandwidth = np.std(bandwidth_arr)
            disc_min, disc_median, disc_max = getDynamicconfig_mpc(
                configmap_mpc.configmap_mpc_oboe_900, mean_bandwidth, std_bandwidth)
            discount = disc_median

        harmonic_bandwidth = np.sum(state[3] * state[4]) / np.sum(state[4])
        future_bandwidth = harmonic_bandwidth * discount
        bit_rate = ccmpc.run(future_bandwidth * 1000.0, buffer_size, last_bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            state = np.zeros((S_INFO, S_LEN))

            throughput = []

            ch_index = -1
            discount = 0.5

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
