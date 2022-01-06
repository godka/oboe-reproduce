import numpy as np
import fixed_env as env
import load_trace
import mpc

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42

def single_step(mu, sigma):

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(mu, sigma)
    reward_all = []
    for mpc_discount in range(10):
        reward_arr = []
        discount = mpc_discount / 10. + 0.1
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)

        ccmpc = mpc.mpc()
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        r_batch = []
        state = np.zeros((S_INFO, S_LEN))
        video_count = 0

        while True:  # serve video forever
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

            r_batch.append(reward)

            last_bit_rate = bit_rate
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K 
            state[4, -1] = float(delay) / M_IN_K 

            thr = state[3]
            delay = state[4]
            harmonic_bandwidth = np.sum(thr * delay) / np.sum(delay) * 1000.
            future_bandwidth = harmonic_bandwidth * discount
            bit_rate = ccmpc.run(future_bandwidth, buffer_size, last_bit_rate)

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                reward_arr.append(np.sum(r_batch))
                del r_batch[:]
                state = np.zeros((S_INFO, S_LEN))
                video_count += 1
                if video_count >= len(all_file_names):
                    break
        reward_all.append(np.array(reward_arr))
    reward_all = np.array(reward_all)
    return np.round(np.argmax(reward_all, axis=0) / 10. + 0.1, 1)


if __name__ == '__main__':
    configmap = {}
    # 0.5mbps - 6mbps
    for mu in range(5, 60):
        for sigma in range(1, 40):
            mu_ = mu / 10.
            sigma_ = sigma / 10.
            best_params = single_step(mu_, sigma_)
            configmap[(int(mu_ * 1000), int(sigma_ * 1000))] = list(best_params)

    f = open('../src/configmap_mpc.py', 'w')
    f.write('configmap_mpc_oboe_900 = ')
    print(configmap, file=f)
    f.close()

    print('done')
    