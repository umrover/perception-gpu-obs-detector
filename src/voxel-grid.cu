#include "voxel-grid.hpp"

using namespace std;

/* --- Kernels --- */
/**
* \brief adds offsets to extrema to make sure all are evenly spaced
* \param extrema: array of min and max indices of points for each axis
* \param pc: GPU point cloud
* \return void
*/
__global__ void makeCubeKernel(pair<float,float>* extrema, GPU_Cloud_F4 pc) {
    if(threadIdx.x >= 6) return; // Only need 6 threads

    // Variable Declarations
    int idx = threadIdx.x;
    int axis = idx/2;
    sl::float4 pt;
    __shared__ float dif[3];
    enum axis{x=0, y=1, z=2};

    // Calculate differences between mins and maxes
    if(idx % 2 == 0) { // If even
        pt = pc.data[(int)extrema[axis].first];
        
        //Find differences between extremes on each axis
        if(axis == 0) dif[axis] = pc.data[(int)extrema[axis].second].x - pt.x;
        else if(axis == 1) dif[axis] = pc.data[(int)extrema[axis].second].y - pt.y;
        else dif[axis] = pc.data[(int)extrema[axis].second].z - pt.z;
    }
    else { // If odd process maxes
        pt = pc.data[(int)extrema[axis].second];
    }

    __syncthreads();

    // Obnoxiously long system for making sure all mins and maxes have same difference
    
    // If z is largest difference add offset to other values
    if(dif[z] >= dif[y] && dif[z] >= dif[x]) {

        if(idx % 2 == 0) { // If even process mins    
            if(axis == x) extrema[axis].first = pt.x - ((dif[z]-dif[x])/2) - 1;

            else if(axis == y) extrema[axis].first = pt.y - ((dif[z]-dif[y])/2) - 1;

            else extrema[axis].first = pt.z - 1;
        }
        else { // If odd process maxes
            if(axis == x) extrema[axis].second = pt.x + ((dif[z]-dif[x])/2) + 1;

            else if(axis == y) extrema[axis].second = pt.y + ((dif[z]-dif[y])/2) + 1;

            else extrema[axis].second = pt.z + 1;
        }
    }

    // If y is largest distance add offset to other values
    else if(dif[y] >= dif[z] && dif[y] >= dif[x]) {
        
        if(idx % 2 == 0) { // If even process mins
            if(axis == x) extrema[axis].first = pt.x - ((dif[y]-dif[x])/2) - 1;

            else if(axis == y) extrema[axis].first = pt.y - 1;

            else extrema[axis].first = pt.z - ((dif[y]-dif[z])/2) - 1;
        }
        else { // If odd process maxes
            if(axis == x) extrema[axis].second = pt.x + ((dif[y]-dif[x])/2) + 1;

            else if(axis == y) extrema[axis].second = pt.y + 1;

            else extrema[axis].second = pt.z + ((dif[y]-dif[z])/2) + 1;
        }
    }

    // If x is largest distance add offset to other values
    else {
        
        if(idx % 2 == 0) { // If even process mins
            if(axis == x) extrema[axis].first = pt.x - 1;

            else if(axis == y) extrema[axis].first = pt.y - ((dif[x]-dif[y])/2) - 1;

            else extrema[axis].first = pt.z - ((dif[x]-dif[z])/2) - 1;
        }
        else { // If odd process maxes
            if(axis == x) extrema[axis].second = pt.x + 1;

            else if(axis == y) extrema[axis].second = pt.y + ((dif[x]-dif[y])/2) + 1;

            else extrema[axis].second = pt.z + ((dif[x]-dif[z])/2) + 1;
        }
    }
    
    return;   
}



/* --- Host Functions --- */

VoxelGrid::VoxelGrid(int partitions) : partitions{partitions} {}

void VoxelGrid::makeBoundingCube(GPU_Cloud_F4 &pc) {
    
    enum axis{x=0, y=1, z=2};
    
    // Create place to store maxes
    thrust::pair< thrust::device_ptr<sl::float4>, thrust::device_ptr<sl::float4>> extrema[3];

    // Find 6 maxes of Point Cloud
    extrema[x] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::X));
    extrema[y] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::Y));
    extrema[z] = thrust::minmax_element(thrust::device_ptr<sl::float4>(pc.data), 
                                        thrust::device_ptr<sl::float4>(pc.data) + pc.size, 
                                        CompareFloat4(Axis::Z));    

    pair<float,float> extremaVals[3] = {
        {extrema[x].first - thrust::device_ptr<sl::float4>(pc.data), extrema[x].second - thrust::device_ptr<sl::float4>(pc.data)},
        {extrema[y].first - thrust::device_ptr<sl::float4>(pc.data), extrema[y].second - thrust::device_ptr<sl::float4>(pc.data)},
        {extrema[z].first - thrust::device_ptr<sl::float4>(pc.data), extrema[z].second - thrust::device_ptr<sl::float4>(pc.data)}
    };

    // Adjust extrema to form a cube
    checkStatus(cudaMalloc(&extremaValsGPU, sizeof(pair<float,float>)*3));
    checkStatus(cudaMemcpy(extremaValsGPU, extremaVals, sizeof(pair<float,float>)*3, cudaMemcpyHostToDevice));

    makeCubeKernel<<<1,MAX_THREADS>>>(extremaValsGPU, pc);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
}
