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

/**
 * \enum Axis
 * \brief Enum for x,y,z axis
 */
enum class Axis {X, Y, Z};

/**
 * \class CompareFloat4
 * \brief Functor that compares Float4 values
*/
class CompareFloat4 {
public: 

    /**
     * \brief CompareFloat4 construct
     * \param axis axis to compare
     */
    CompareFloat4(Axis axisIn) : axis{axisIn} {}

    /**
     * \brief overloaded operator for comparing if lhs < rhs on given axis
     * \param lhs: float to compare
     * \param rhs: float to compare
     * \return bool
     */
    __host__ __device__ bool operator() (sl::float4 lhs, sl::float4 rhs) {
        
        switch (axis) {
        
            case Axis::X :
                return lhs.x < rhs.x;
        
            case Axis::Y :
                return lhs.y < rhs.y;
        
            case Axis::Z :
                return lhs.z < rhs.z;
        }
    };

private:

    Axis axis;

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

__device__ float atomicMinFloat (float* addr, float value);

__device__ float atomicMaxFloat (float* addr, float value);

#define MAX_THREADS 1024


#endif