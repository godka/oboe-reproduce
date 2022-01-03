import numpy as np
import fixed_env as env
import load_trace
import matplotlib.pyplot as plt
import itertools
import configmap_hyb_oboe
import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './test_results'
LOG_FILE = SUMMARY_DIR + '/log_sim_oboehyb'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth

#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075,
               2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990,
               1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175,
               897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667,
               587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800,
               359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654,
               146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

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

def getDynamicconfig_hyb(pv_list_hyb, bw, std, step=900):
    bw_step = step
    std_step = step
    ABRAlgo = ''
    bw_cut =int(float(bw)/bw_step)*bw_step
    std_cut = int(float(std)/std_step)*std_step
    abr_list = list()
    current_list_1 = list()
    current_list_2 = list()
    current_list_bb_1 = list()
    current_list_bb_2 = list()
    current_list_hyb = list()
    count = 0
    #if combination == True:
    if True:
        if bw==-1 and std==-1:
            return 'HYB', 0.25, 0.25, 0.25, 5, 5, 5, 0.4, 0.4, 0.4
        # if key not in performance vector
        if (bw_cut, std_cut) not in list(pv_list_hyb.keys()):
            for i in range(2, 1000, 1):
                count += 1
                for bw_ in [bw_cut - (i - 1) * bw_step, bw_cut + (i-1) * bw_step]:
                    for std_ in range(std_cut - (i - 1) * std_step, std_cut + (i-1) * std_step + std_step, std_step):
                        if (bw_, std_) in list(pv_list_hyb.keys()):
                            #abr_list = abr_list + ABRs[(bw_, std_)]
                            #current_list_bb_1 = current_list_bb_1 + pv_list_bb_1[(bw_, std_)]
                            #current_list_bb_2 = current_list_bb_2 + pv_list_bb_2[(bw_, std_)]
                            current_list_hyb = current_list_hyb + pv_list_hyb[(bw_, std_)]
                for std_ in [std_cut - (i - 1) * std_step, std_cut + (i-1) * std_step]:
                    for bw_ in range(bw_cut - (i - 2) * bw_step, bw_cut + (i-1) * bw_step, bw_step):
                        if (bw_, std_) in list(pv_list_hyb.keys()):
                            #abr_list = abr_list + ABRs[(bw_, std_)]
                            #current_list_bb_1 = current_list_bb_1 + pv_list_bb_1[(bw_, std_)]
                            #current_list_bb_2 = current_list_bb_2 + pv_list_bb_2[(bw_, std_)]
                            current_list_hyb = current_list_hyb + pv_list_hyb[(bw_, std_)]
                if len(current_list_hyb)==0:
                    continue
                else:# len(abr_list)>0 and 'BB' not in abr_list:
                    ABRAlgo = 'HYB'
                    #print "HYB", bw_cut, std_cut, count, sys.argv[1]
                    break
        else:
            #abr_list = ABRs[(bw_cut, std_cut)]
            #current_list_bb_1 = current_list_bb_1 + pv_list_bb_1[(bw_cut, std_cut)]
            #current_list_bb_2 = current_list_bb_2 + pv_list_bb_2[(bw_cut, std_cut)]
            current_list_hyb = current_list_hyb + pv_list_hyb[(bw_cut, std_cut)]
            ABRAlgo = 'HYB'


    #if combination ==True:
    if len(current_list_hyb)==0:
        return 0.25, 0.25, 0.25, 5, 5, 5, 0.4, 0.4, 0.4
    #return ABRAlgo, min(current_list_hyb), statistics.median(current_list_hyb), max(current_list_hyb), 0,0,0,0,0,0
    return min(current_list_hyb), np.percentile(current_list_hyb,10), max(current_list_hyb), 0,0,0,0,0,0

def get_chunk_size(quality, index):
    if (index < 0 or index > 48):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index],
             2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        './cooked_test_traces/')

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    past_errors = []
    past_buffer_errors = []
    past_bandwidth_ests = []
    past_buffer_ests = []
    throughput = []

    video_count = 0
    ch_index = -1
    beta = 0.25
    discount = None
    # make chunk combination options
    for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
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

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(past_bandwidth_ests) > 0):
            curr_error = abs(
                past_bandwidth_ests[-1]-state[3, -1])/float(state[3, -1])
        past_errors.append(curr_error)
        throughput.append(state[3, -1])
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(past_buffer_ests) > 0):
            curr_error = abs(
                past_buffer_ests[-1] - (buffer_size - 4.))/float(buffer_size - 4. + 1e-6)
        past_buffer_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        mean_bandwidth = np.mean(past_bandwidths) * 1000. * 8.
        std_bandwidth = np.std(past_bandwidths) * 1000. * 8.
        
        ch_detected, ch_idx = onlineCD(ch_index, np.array(throughput) * 8000.)
        if ch_detected:
            ch_index = ch_idx
            p1_min, p1_median, p1_max, p2_min, p2_median, p2_max,p3_min, p3_median, p3_max = getDynamicconfig_hyb(
                configmap_hyb_oboe.configmap_hyb_oboe_900, mean_bandwidth, std_bandwidth)
            beta = p1_min
            
        _index = 0.
        for p in range(len(VIDEO_BIT_RATE)):
            #_bit_rate = VIDEO_BIT_RATE[-1 - p] / 8. / 1024.
            _size = next_video_chunk_sizes[-1 - p] / M_IN_K / M_IN_K
            _buff = buffer_size * beta
            _total_capa = _buff * harmonic_bandwidth
            if _size < _total_capa:
                _index = len(VIDEO_BIT_RATE) - 1 - p
                break
        bit_rate = int(_index)

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            throughput = []
            past_errors = []
            past_buffer_errors = []
            past_bandwidth_ests = []
            past_buffer_ests = []

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            ch_index = -1
            beta = 0.25

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
