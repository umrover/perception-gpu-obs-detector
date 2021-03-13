#include "voxel-grid.hpp"

using namespace std;

VoxelGrid::VoxelGrid(int partitions) : partitions{partitions} {}

__global__ void compareKernel(GPU_Cloud_F4 pc, CompareFloat4* compare) {
    if(threadIdx.x == 0) {
        printf("Check %i\n", pc.size);
        printf("1.x: %f\n", pc.data->x);
        printf("2.x: %f\n", (pc.data+pc.size-1)->x);
        printf("Test: %i\n", (*compare)(pc.data[0],pc.data[1] ));
    }
        
        
    
}

void VoxelGrid::buildBins(GPU_Cloud_F4 &pc) {
    
    CompareFloat4 xCompare(Axis::X);
    CompareFloat4* funcPtr;
    cudaMalloc(&funcPtr, sizeof(CompareFloat4));
    printf("uh\n");
    checkStatus(cudaGetLastError());
    cudaMemcpy(funcPtr, &xCompare,  sizeof(CompareFloat4), cudaMemcpyHostToDevice);
    checkStatus(cudaGetLastError());
    printf("uh2\n");
    printf("Ok\n");
    compareKernel<<<1,1024>>>(pc,funcPtr);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    //printf("Test3: %i\n", funcPtr->operator()(pc.data[0], pc.data[1]));
    printf("Passed %i\n", pc.size);
    thrust::pair<sl::float4*,sl::float4*> extrema = thrust::minmax_element(pc.data, pc.data + pc.size, *funcPtr);
    checkStatus(cudaGetLastError());
    printf("Min: %d\n", *extrema.first);
    printf("Max: %d\n", *extrema.second);

}
