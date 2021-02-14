#include "common.hpp"
#include <iostream>
#include <sl/Camera.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>


#include <pcl/common/time.h>

#include <vector>
#include <algorithm>

//Cuda error checking function
bool checkStatus(cudaError_t status) {
	if (status != cudaSuccess) {
		printf("%s \n", cudaGetErrorString(status));
		return true;
	}
    return false;
}

//ceiling division
int ceilDiv(int x, int y) {
    return (x + y - 1) / y;
}


//This function convert a RGBA color packed into a packed RGBA PCL compatible format
inline float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}

//Taken from mrover code, creates a PCL pointcloud from a zed GPU cloud
/*
void ZedToPcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & p_pcl_point_cloud, sl::Mat zed_cloud) {
  sl::Mat zed_cloud_cpu;
  zed_cloud.copyTo(zed_cloud_cpu,  sl::COPY_TYPE::GPU_CPU);
 
  float* p_data_cloud = zed_cloud_cpu.getPtr<float>();
  int index = 0;
  for (auto &it : p_pcl_point_cloud->points) {
    float X = p_data_cloud[index];
    if (!isValidMeasure(X)) // Checking if it's a valid point
        it.x = it.y = it.z = it.rgb = 0;
    else {
        it.x = X;
        it.y = p_data_cloud[index + 1];
        it.z = p_data_cloud[index + 2];
        it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
    }
    index += 4;
  }

} */

GPU_Cloud getRawCloud(sl::Mat zed_cloud) {
    GPU_Cloud g;
    g.data = zed_cloud.getPtr<float>(sl::MEM::GPU);
    g.stride = 4;
    g.size = zed_cloud.getWidth() * zed_cloud.getHeight();
    return g;
}


GPU_Cloud_F4 getRawCloud(sl::Mat zed_cloud, bool f4) {
    GPU_Cloud_F4 g;
    g.data = zed_cloud.getPtr<sl::float4>(sl::MEM::GPU);
    g.stride = 4;
    g.size = zed_cloud.getWidth() * zed_cloud.getHeight();
    return g;
}

GPU_Cloud_F4 createCloud(int size) {
    GPU_Cloud_F4 g;
    cudaMalloc(&g.data, sizeof(sl::float4)*size);
    g.stride = 4;
    g.size = size;
    return g;
}


__global__ void copyKernel(GPU_Cloud_F4 to, GPU_Cloud_F4 from) {
    int pointIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pointIdx >= from.size) return;
    to.data[pointIdx] = from.data[pointIdx];
}

__global__ void removeJunkKernel(GPU_Cloud_F4 cloud, int start, int maxSize) {
    int pointIdx = start + threadIdx.x + blockIdx.x * blockDim.x;
    if(pointIdx >= maxSize) return;
    cloud.data[pointIdx].x = 0;
    cloud.data[pointIdx].y = 0;
    cloud.data[pointIdx].z = 0;
    cloud.data[pointIdx].w = VIEWER_BGR_COLOR;

}

void copyCloud(GPU_Cloud_F4 &to, GPU_Cloud_F4 &from) {
    to.size = from.size;
    copyKernel<<<ceilDiv(from.size, MAX_THREADS), MAX_THREADS>>>(to, from);
    checkStatus(cudaDeviceSynchronize());
}

void clearStale(GPU_Cloud_F4 &cloud, int maxSize) {
    removeJunkKernel<<<ceilDiv(maxSize-cloud.size, MAX_THREADS), MAX_THREADS>>>(cloud, cloud.size, maxSize);
    checkStatus(cudaDeviceSynchronize());
}