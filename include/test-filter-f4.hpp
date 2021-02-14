#include <sl/Camera.hpp>
#include "common.hpp"


#ifndef TEST_FILTER_F4
#define TEST_FILTER_F4


class TestFilter_F4 {

    public:
        //Initialized filter with point cloud
        TestFilter_F4(sl::Mat gpu_cloud);
        
        //Run the filter
        void run();

    private:
        GPU_Cloud_F4 gpu_cloud;

};

#endif