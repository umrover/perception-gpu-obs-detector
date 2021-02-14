#include "pass-through.hpp"
#include <stdlib.h>
#include <cmath>
#include <limits>
#include "common.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>

PassThrough::PassThrough(char axis, float min, float max) : min{min}, max{max}, axis(axis){

    //Set the axis value


};

#ifdef OLD
/*
//CUDA Kernel Helper Function
__global__ void passThroughKernel(GPU_Cloud_F4 cloud, GPU_Cloud_F4 out, int axis, float min, float max, int* size) {

    //Find index for current operation
    int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    if(idx > cloud.size)
        return;

    //If out of range make blue and return
    if(     (axis == 0 && (cloud.data[idx].x > max || cloud.data[idx].x < min ||
            isnan(cloud.data[idx].x) || isinf(cloud.data[idx].x)))

        ||  (axis == 1 && (cloud.data[idx].y > max || cloud.data[idx].y < min ||
            isnan(cloud.data[idx].y) || isinf(cloud.data[idx].y)))

        || (axis == 2 && (cloud.data[idx].z > max || cloud.data[idx].z < min ||
            isnan(cloud.data[idx].z) || isinf(cloud.data[idx].z)))
    ) {
        //cloud.data[idx].w = 4353.0;
        //zed viewer background
        cloud.data[idx].w = 2.35098856151e-38; //VIEWER_BGR_COLOR;//


        return;
    }
       
    //If still going then update point cloud float array
    sl::float4 copy = cloud.data[idx];

    //Make sure all threads have checked for passThrough
    __syncthreads();

    //Count the new size
    int place = atomicAdd(size, 1);

    //Copy back data into place in front of array
    out.data[place] = copy;

}

void PassThrough::run(GPU_Cloud_F4 &cloud){
    GPU_Cloud_F4 tmpCloud = createCloud(cloud.size); //exp

    std::cerr << "Original size: " << cloud.size << "\n";
    
    //Create pointer to value in host memory
    int* h_newSize = new int;
    *h_newSize = 0;

    //Create pointer to value in device memory
    int* d_newSize = nullptr;
    cudaMalloc(&d_newSize, sizeof(int));

    //Copy from host to device
    cudaMemcpy(d_newSize, h_newSize, sizeof(int), cudaMemcpyHostToDevice);
    
    //Run PassThrough Kernel
    passThroughKernel<<<ceilDiv(cloud.size, BLOCK_SIZE), BLOCK_SIZE>>>(cloud, tmpCloud, axis, min, max, d_newSize);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    
    //Copy from device to host
    cudaMemcpy(h_newSize, d_newSize, sizeof(int), cudaMemcpyDeviceToHost);
    
    //Update size of cloud
    tmpCloud.size = *h_newSize;
    copyCloud(cloud, tmpCloud); //exp
    cudaFree(tmpCloud.data); //exp

    std::cerr << "New Cloud Size: " << cloud.size << "\n";

    //Free dynamically allocated memory
    cudaFree(d_newSize);
    delete h_newSize;
}
*/
#endif

//Functor predicate to check if a point is within some min and max bounds on a particular axis
class WithinBounds {
    public:
        WithinBounds(float min, float max, char axis) : min(min), max(max), axis(axis) {}

        __host__ __device__ bool operator()(const sl::float4 val) {
            float test;
            if(axis == 'z') test = val.z;
            else if(axis == 'y') test = val.y;
            return test > min && test < max;
        }

    private:
        float min, max;
        char axis;
};

//Execute pass through
void PassThrough::run(GPU_Cloud_F4 &cloud){

    //Instansiate a predicate functor and copy the contents of the cloud
    WithinBounds pred(min, max, axis);
    thrust::device_vector<sl::float4> buffer(cloud.data, cloud.data+cloud.size);

    //Copy from the temp buffer back into the cloud only the points that pass the predicate 
    sl::float4* end = thrust::copy_if(thrust::device, buffer.begin(), buffer.end(), cloud.data, pred);

    //Clear the remainder of the cloud of points that failed pass through
    thrust::fill(thrust::device, end, cloud.data+cloud.size, sl::float4(0, 0, 0, 0));

    //update the cloud size
    cloud.size = end - cloud.data;
}

