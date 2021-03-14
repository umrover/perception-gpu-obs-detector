#include "voxel-grid.hpp"

using namespace std;

VoxelGrid::VoxelGrid(int partitions) : partitions{partitions} {}

__global__ void adjustBoxKernel(pair<float,float>* extrema, GPU_Cloud_F4 pc) {
    if(threadIdx.x >= 6) return; // Only need 6 threads

    int idx = threadIdx.x + 1;
    sl::float4 pt;
    __shared__ float dif[3];
    enum axis{x=0, y=1, z=2};

    if(idx % 2 == 0) { // If even process the mins and difference
        pt = pc.data[extrema[idx/3].first];
        
        //Find differences between extremes on each axis
        if(idx/3 == 0) dif[idx/3] = pc.data[extrema[idx/3].second].x - pt.x;
        else if(idx/3 == 1) dif[idx/3] = pc.data[extrema[idx/3].second].y - pt.y;
        else dif[idx/3] = pc.data[extrema[idx/3].second].z - pt.z;
    }
    else { // If odd process maxes
        pt = pc.data[extrema[idx/3].second];
    }

    __syncthreads();

    // Obnoxiously long system for making sure all mins and maxes have same difference
    
    // If z is largest difference add offset to other values
    if(dif[z] >= dif[y] && dif[z] >= dif[x]) {

        if(idx % 2 == 0) { // If even process mins    
            if(idx/3 == x) extrema[idx/3].first = pt.x - (dif[z]-dif[x]/2) - 1;

            else if(idx/3 == y) extrema[idx/3].first = pt.y - (dif[z]-dif[y]/2) - 1;

            else extrema[idx/3].first = pt.z - 1
        }
        else { // If odd process maxes
            if(idx/3 == x) extrema[idx/3].second = pt.x + (dif[z]-dif[x]/2) + 1;

            else if(idx/3 == y) extrema[idx/3].second = pt.y + (dif[z]-dif[y]/2) + 1;

            else extrema[idx/3].second = pt.z + 1;
        }
    }

    // If y is largest distance add offset to other values
    else if(dif[y] >= dif[z] && dif[y] >= dif[x]) {
        
        if(idx % 2 == 0) { // If even process mins
            if(idx/3 == x) extrema[idx/3].first = pt.x - (dif[y]-dif[x]/2) - 1;

            else if(idx/3 == y) extrema[idx/3].first = pt.y - 1;

            else extrema[idx/3].first = pt.z - (dif[y]-dif[z]/2) - 1
        }
        else { // If odd process maxes
            if(idx/3 == x) extrema[idx/3].second = pt.x + (dif[y]-dif[x]/2) + 1;

            else if(idx/3 == y) extrema[idx/3].second = pt.y + 1;

            else extrema[idx/3].second = pt.z + (dif[y]-dif[z]/2) + 1;
        }
    }

    // If x is largest distance add offset to other values
    else {
        
        if(idx % 2 == 0) { // If even process mins
            if(idx/3 == x) extrema[idx/3].first = pt.x - 1;

            else if(idx/3 == y) extrema[idx/3].first = pt.y - (dif[x]-dif[y]/2) - 1;

            else extrema[idx/3].first = pt.z - (dif[x]-dif[z]/2) - 1
        }
        else { // If odd process maxes
            if(idx/3 == x) extrema[idx/3].second = pt.x + 1;

            else if(idx/3 == y) extrema[idx/3].second = pt.y + (dif[x]-dif[y]/2) - 1;

            else extrema[idx/3].second = pt.z + (dif[x]-dif[z]/2) + 1;
        }
    }

    return;   
}

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

    pair<float,float>* extremaVals = {
        {extrema[x].first - thrust::device_ptr<sl::float4>(pc.data), extrema[x].second - thrust::device_ptr<sl::float4>(pc.data)},
        {extrema[y].first - thrust::device_ptr<sl::float4>(pc.data), extrema[y].second - thrust::device_ptr<sl::float4>(pc.data)},
        {extrema[z].first - thrust::device_ptr<sl::float4>(pc.data), extrema[z].second - thrust::device_ptr<sl::float4>(pc.data)}
    };

    printf("MinX: %i\n", extrema[x].first - thrust::device_ptr<sl::float4>(pc.data));
    printf("MaxX: %i\n", extrema[x].second - thrust::device_ptr<sl::float4>(pc.data));
    printf("MinY: %i\n", extrema[y].first - thrust::device_ptr<sl::float4>(pc.data));
    printf("MaxY: %i\n", extrema[y].second - thrust::device_ptr<sl::float4>(pc.data));
    printf("MinZ: %i\n", extrema[z].first - thrust::device_ptr<sl::float4>(pc.data));
    printf("MaxZ: %i\n", extrema[z].second - thrust::device_ptr<sl::float4>(pc.data));

}
