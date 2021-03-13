#include "voxel-grid.hpp"

using namespace std;

VoxelGrid::VoxelGrid(int partitions) : partitions{partitions} {}

void VoxelGrid::buildBins(GPU_Cloud_F4 &pc) {
    
    enum axis{x=0, y=1, z=2};
    
    //Create place to store maxes
    thrust::pair< thrust::device_ptr<sl::float4>, thrust::device_ptr<sl::float4>> extrema[3];

    //Find 6 maxes of Point Cloud
    extrema[x] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::X));
    extrema[y] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::Y));
    extrema[z] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::Z));    
                                                                
    printf("Min: %i\n", extrema[x].first - thrust::device_ptr<sl::float4>(pc.data));
    printf("Max: %i\n", extrema[x].second - thrust::device_ptr<sl::float4>(pc.data));

}
