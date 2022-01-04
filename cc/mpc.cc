#include <iostream>
#include <stdlib.h>
#include <iostream>
#include "mpc.h"
#define MPC_FUTURE_CHUNK_COUNT 5
#define A_DIM 6

mpc::mpc()
{

    for (auto idx = 0; idx < std::pow(A_DIM, MPC_FUTURE_CHUNK_COUNT); idx++)
    {
        std::vector<int> vec;
        int j = idx;
        for (auto i = 0; i < MPC_FUTURE_CHUNK_COUNT; ++i)
        {
            auto tmp = j % A_DIM;
            vec.push_back(tmp);
            j /= A_DIM;
        }
        CHUNK_COMBO_OPTIONS.push_back(vec);
    }
}

int mpc::run(double future_bandwidth, double buffer_size, int last_video_bitrate)
{
    double max_reward = -100000000;
    int send_data = 0;
    auto start_buffer = buffer_size;
    for (auto &combo : CHUNK_COMBO_OPTIONS)
    {
        double curr_buffer = start_buffer;
        double curr_rebuffer_time = 0.0;
        double bitrate_sum = 0.0;
        double bitrate_smoothness = 0.0;
        double bitrate_last = last_video_bitrate;
        double reward_ = 0.0;
        
        for (auto position = 0; position < MPC_FUTURE_CHUNK_COUNT; position++)
        {
            auto chunk_quality = combo[position];
            auto size = VIDEO_BIT_RATE[chunk_quality] * 4. / 8.; // KB
            auto download_time = size / future_bandwidth; // KB/s
            //std::cout << size << " " << future_bandwidth << " " << download_time << std::endl;
            download_time += 80.0 / 1000.0;
            //double curr_buffer = 0.0;
            if (curr_buffer < download_time)
            {
                curr_rebuffer_time += (download_time - curr_buffer);
                curr_buffer = 0.0;
            }
            else
            {
                curr_buffer -= download_time;
            }
            curr_buffer += 4.0;
            auto bitrate_current = VIDEO_BIT_RATE[chunk_quality];
            bitrate_sum += bitrate_current;
            bitrate_smoothness += std::abs(bitrate_current - VIDEO_BIT_RATE[bitrate_last]);
            bitrate_last = chunk_quality;

            reward_ = bitrate_sum / 1000.0 - 4.3 * curr_rebuffer_time - bitrate_smoothness / 1000.0;
        }
        if (reward_ >= max_reward)
        {
            max_reward = reward_;
            send_data = combo[0];
        }
    }
    // std::cout << send_data << std::endl;
    return send_data;
}

PYBIND11_MODULE(mpc, m) {
    pybind11::class_<mpc>(m, "mpc")
        .def(pybind11::init<>())
        .def("run", &mpc::run);
}