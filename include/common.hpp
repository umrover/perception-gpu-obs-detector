#include <sl/Camera.hpp>

#ifndef COMMON
#define COMMON

#define CHANNEL 4
#define BLOCK_SIZE 1024
#define PI 3.141592
#define HALF_ROVER 584
#define VIEWER_BGR_COLOR 2.14804915479e-38


//GPU point cloud struct that can be passed to cuda kernels and represents a point cloud
struct GPU_Cloud_F4 {
    sl::float4* data;
    int stride; 
    int size;
};

//GPU Indicies data
struct GPU_Indicies {
    int* data;
    int size;
};


//Returns true if a cuda error occured and prints an error message
bool checkStatus(cudaError_t status);

//ceiling division x/y. e.g. ceilDiv(3,2) -> 2
int ceilDiv(int x, int y);

//Get a CUDA workable gpu point cloud struct from Zed GPU cloud
GPU_Cloud_F4 getRawCloud(sl::Mat zed_cloud);

GPU_Cloud_F4 createCloud(int size);

void copyCloud(GPU_Cloud_F4 &to, GPU_Cloud_F4 &from);

void clearStale(GPU_Cloud_F4 &cloud, int maxSize);

__global__ void findClearPathKernel(float* minXG, float* maxXG, float* minZG, float* maxZ, int numClusters, int* leftBearing, int* rightBearing);

__global__ void findAngleOffCenterKernel(float* minXG, float* maxXG, float* minZG, float* maxZ, int numClusters, int* bearing, int direction);


#define MAX_THREADS 1024


#endif