#include <deque>
#include <fstream>
#include <tuple>
#include <numeric>
#include <random>
#include <pybind11/pybind11.h>

class mpc
{
    public:
        mpc();
        int run(double throughput, double buffer, int last_video_bitrate);
    private:
        std::vector<std::vector<int>> CHUNK_COMBO_OPTIONS;
        std::vector<int> VIDEO_BIT_RATE = {300, 750, 1200, 1850, 2850, 4300}; //Kbps
        
};