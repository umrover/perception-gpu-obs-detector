#include "euclidean-cluster.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "common.hpp"
#include <limits>     //for std::numeric_limits<> 

//Helper functions
__device__ float getFloatData(int axis, sl::float4 &val) {
    if(!axis)
        return val.x;
    else if(axis == 1)
        return val.y;
    else
        return val.z;
}
            
__device__ float getData(int axis, int index, sl::float4 *data) {
    return getFloatData(axis, data[index]);    
}

//Hash function that deteremines bin number
__device__ int hashToBin(sl::float4 &data, float* min, float* max, int partitions) {
    int cpx = (data.x-min[0])/(max[0]-min[0])*partitions;
    int cpy = (data.y-min[1])/(max[1]-min[1])*partitions;
    int cpz = (data.z-min[2])/(max[2]-min[2])*partitions;
    return cpx*partitions*partitions+cpy*partitions+cpz;
}


/**
This kernel uses parallel reduction to find the 6 maximum and minimum points
in the point cloud
*/
__global__ void findBoundingBoxKernel(GPU_Cloud_F4 pc, int *minXGlobal, int *maxXGlobal,
                                int *minYGlobal, int *maxYGlobal, int *minZGlobal, int *maxZGlobal){
    //Would it be better to do multiple parallel reductions than one large memory consuming reduction?
    //This method makes 6 copies of the point cloud to find the necessary values 
    const int threads = MAX_THREADS;
    __shared__ int localMinX[threads/2];
    __shared__ int localMaxX[threads/2];
    __shared__ int localMinY[threads/2];
    __shared__ int localMaxY[threads/2];
    __shared__ int localMinZ[threads/2];
    __shared__ int localMaxZ[threads/2];
    __shared__ sl::float4 data[threads];
    __shared__ bool notFull;

    sl::float4 defaultInit(-1.0,-1.0 , -1.0, 0);

    int actualIndex = threadIdx.x + blockIdx.x * blockDim.x;
    

    if(actualIndex < pc.size){ //only write to shared memory if threads about to die
        data[threadIdx.x] = pc.data[actualIndex]; //Write from global memory into shared memory
    }
    else { //Accounts for final block with more threads than points
        notFull = true;
        data[threadIdx.x] = defaultInit;
    }
    __syncthreads();

    int aliveThreads = threads / 2;

    if(!notFull) { //Don't have to worry about checking for going out of bounds
    
        int minX = threadIdx.x, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minZ; //initialize local indices of mins and maxes
        
        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            minX = (data[aliveThreads + threadIdx.x].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
            maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x) ? aliveThreads + threadIdx.x : maxX;
            minY = (data[aliveThreads + threadIdx.x].y < data[minY].y) ? aliveThreads + threadIdx.x : minY;
            maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y) ? aliveThreads + threadIdx.x : maxY;
            minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z) ? aliveThreads + threadIdx.x : minZ;
            maxZ = (data[aliveThreads + threadIdx.x].z > data[maxZ].z) ? aliveThreads + threadIdx.x : maxZ;
            if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                localMinX[threadIdx.x] = minX;
                localMaxX[threadIdx.x] = maxX;
                localMinY[threadIdx.x] = minY;
                localMaxY[threadIdx.x] = maxY;
                localMinZ[threadIdx.x] = minZ;
                localMaxZ[threadIdx.x] = maxZ;
            }
        }
        __syncthreads();
        aliveThreads /= 2;

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? localMinX[aliveThreads + threadIdx.x] : minX;
                maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? localMaxX[aliveThreads + threadIdx.x] : maxX;
                minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? localMinY[aliveThreads + threadIdx.x] : minY;
                maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? localMaxY[aliveThreads + threadIdx.x] : maxY;
                minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? localMinZ[aliveThreads + threadIdx.x] : minZ;
                maxZ = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z) ? localMaxZ[aliveThreads + threadIdx.x] : maxZ;
                if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                    localMinX[threadIdx.x] = minX;
                    localMaxX[threadIdx.x] = maxX;
                    localMinY[threadIdx.x] = minY;
                    localMaxY[threadIdx.x] = maxY;
                    localMinZ[threadIdx.x] = minZ;
                    localMaxZ[threadIdx.x] = maxZ;
                }
            }
            __syncthreads();
            aliveThreads /= 2;
        }
    }
    
    else{ //Now we have to worry ab having a block that's not full
        int minX = actualIndex < pc.size ? threadIdx.x : -1, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minX; //initialize local indices of mins and maxes checking for those exceeding size
        
        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            
            if(aliveThreads + threadIdx.x + blockDim.x*blockIdx.x < pc.size) { //If points to valid data
                minX = (data[aliveThreads + threadIdx.x].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x) ? aliveThreads + threadIdx.x : maxX;
                minY = (data[aliveThreads + threadIdx.x].y < data[minY].y) ? aliveThreads + threadIdx.x : minY;
                maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y) ? aliveThreads + threadIdx.x : maxY;
                minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z) ? aliveThreads + threadIdx.x : minZ;
                maxZ = (data[aliveThreads + threadIdx.x].z > data[maxZ].z) ? aliveThreads + threadIdx.x : maxZ;
            }
            if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                localMinX[threadIdx.x] = minX;
                localMaxX[threadIdx.x] = maxX;
                localMinY[threadIdx.x] = minY;
                localMaxY[threadIdx.x] = maxY;
                localMinZ[threadIdx.x] = minZ;
                localMaxZ[threadIdx.x] = maxZ;
            }
        }
        __syncthreads();
        aliveThreads /= 2;

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                if(localMinX[aliveThreads + threadIdx.x] >= 0) { //If valid value compare and choose appropriately
                    if(data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) minX = localMinX[aliveThreads + threadIdx.x];
                
                    //minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                    maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? localMaxX[aliveThreads + threadIdx.x] : maxX;
                    minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? localMinY[aliveThreads + threadIdx.x] : minY;
                    maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? localMaxY[aliveThreads + threadIdx.x] : maxY;
                    minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? localMinZ[aliveThreads + threadIdx.x] : minZ;
                    maxZ = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z) ? localMaxZ[aliveThreads + threadIdx.x] : maxZ;
                }
                if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                    localMinX[threadIdx.x] = minX;
                    localMaxX[threadIdx.x] = maxX;
                    localMinY[threadIdx.x] = minY;
                    localMaxY[threadIdx.x] = maxY;
                    localMinZ[threadIdx.x] = minZ;
                    localMaxZ[threadIdx.x] = maxZ;
                }
            }
            __syncthreads();
            aliveThreads /= 2;
        }
    }
    //Write to global memory
    if(threadIdx.x == 0){
        minXGlobal[blockIdx.x] = localMinX[0] + blockDim.x*blockIdx.x;
        maxXGlobal[blockIdx.x] = localMaxX[0] + blockDim.x*blockIdx.x;
        minYGlobal[blockIdx.x] = localMinY[0] + blockDim.x*blockIdx.x;
        maxYGlobal[blockIdx.x] = localMaxY[0] + blockDim.x*blockIdx.x;
        minZGlobal[blockIdx.x] = localMinZ[0] + blockDim.x*blockIdx.x;
        maxZGlobal[blockIdx.x] = localMaxZ[0] + blockDim.x*blockIdx.x;
    }
    return;

}

/**
The final reduction to extrema to find the ultimate extrema from the
provided list. Split into 3 blocks each calculating the max and min 
values for their given axis. Needed to divide it up since float4 = 16bytes
and we have 2048 float4
*/
__global__ void findExtremaKernel (GPU_Cloud_F4 pc, int size, int *minGlobal, int *maxGlobal, 
    float* finalMin, float* finalMax, int axis) {
    
    //Copy from global to shared memory
    const int threads = MAX_THREADS;
    __shared__ int localMin[threads];
    __shared__ int localMax[threads];
    __shared__ sl::float4 localMinData[threads];
    __shared__ sl::float4 localMaxData[threads];
    
    //Copy in all of the local data check for uninitialized values
    //Shouldn't cause warp divergence since the first set of contiguous
    //numbers will enter the else and the second half will enter the if
    
    if(threadIdx.x >= size) {
        localMin[threadIdx.x] = -1;
        localMax[threadIdx.x] = -1;
        localMinData[threadIdx.x] = pc.data[0];
        localMaxData[threadIdx.x] = pc.data[0];
    }
    else {
        localMin[threadIdx.x] = minGlobal[threadIdx.x];
        localMax[threadIdx.x] = maxGlobal[threadIdx.x];
        localMinData[threadIdx.x] = pc.data[localMin[threadIdx.x]];
        localMaxData[threadIdx.x] = pc.data[localMax[threadIdx.x]];
    }
    __syncthreads();

    //Registry memory initializations
    int min = localMin[threadIdx.x];
    int max = localMax[threadIdx.x];
    int aliveThreads = (blockDim.x) / 2;
    sl::float4 minData = localMinData[threadIdx.x];
    sl::float4 maxData = localMaxData[threadIdx.x];

    __syncthreads();

    //Do parallel reduction and modify both values as you go along
    while (aliveThreads > 0) {
        if (threadIdx.x < aliveThreads && localMin[threadIdx.x+aliveThreads] != -1) {
            //Check if value smaller than min
            if(getFloatData(axis, minData) > getFloatData(axis, localMinData[threadIdx.x + aliveThreads])) {
                minData = localMinData[threadIdx.x + aliveThreads];
                min = localMin[threadIdx.x + aliveThreads];
            }
            //Check if value larger than max
            if(getFloatData(axis, maxData) < getFloatData(axis, localMaxData[threadIdx.x + aliveThreads])) {
                maxData = localMaxData[threadIdx.x + aliveThreads];
                max = localMax[threadIdx.x + aliveThreads];
            }

            //Check if thread is going to die next iteration
            if (threadIdx.x >= (aliveThreads) / 2) {
                localMin[threadIdx.x] = min;
                localMax[threadIdx.x] = max;
                localMinData[threadIdx.x] = minData;
                localMaxData[threadIdx.x] = maxData;
            }
        }
        __syncthreads();
        aliveThreads /= 2;
    }

    //If final thread write to global memory
    if(threadIdx.x == 0){
        finalMin[axis] = getFloatData(axis, minData);
        finalMax[axis] = getFloatData(axis, maxData);
        //std::printf("Axis %i min index: %.1f\n", axis, getFloatData(axis, localMinData[threadIdx.x]));
        //std::printf("Axis %i max index: %.1f\n", axis, getFloatData(axis, localMaxData[threadIdx.x]));
        
        //If the last axis calculated readjust so the values make a cube
        if(axis == 2){
            float difX = finalMax[0]-finalMin[0];
            float difY = finalMax[1]-finalMin[1];
            float difZ = finalMax[2]-finalMin[2];
    
            if(difZ >= difY && difZ >= difX) {
                float addY = (difZ-difY)/2+1;
                float addX = (difZ-difX)/2+1;
                finalMax[0] += addX;
                finalMin[0] -= addX;
                finalMax[1] += addY;
                finalMin[1] -= addY; 
                finalMax[2] += 1;
                finalMin[2] -= 1;
            }

            else if(difY >= difX && difY >= difZ) {
                float addZ = (difY-difZ)/2+1;
                float addX = (difY-difX)/2+1;
                finalMax[0] += addX;
                finalMin[0] -= addX;
                finalMax[2] += addZ;
                finalMin[2] -= addZ;
                finalMax[1] += 1;
                finalMin[1] -= 1;
            }

            else {
                float addY = (difX-difY)/2+1;
                float addZ = (difX-difZ)/2+1;
                finalMax[2] += addZ;
                finalMin[2] -= addZ;
                finalMax[1] += addY;
                finalMin[1] -= addY;
                finalMax[0] += 1;
                finalMin[0] -= 1;
            }

        }
    }
      
}

/*
Finds the 6 maximum and minimum points needed to define a bounding box around the 
point cloud. Performs a function 6 times to find each point. The maximum pc size
for this function is 1048576 since it assumes the resulting reduction fits into a block
*/
void EuclideanClusterExtractor::findBoundingBox(GPU_Cloud_F4 &pc){
    const int threads = MAX_THREADS;
    int blocks = ceilDiv(pc.size,threads);
    int *minX; //Stores max and min x,y,z values for each block in global memory
    int *maxX;
    int *minY; 
    int *maxY;
    int *minZ; 
    int *maxZ;

    std::cerr << "Find bound\n";
    checkStatus(cudaMalloc(&minX, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxX, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&minY, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxY, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&minZ, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxZ, sizeof(int) * blocks));

    //Find 6 bounding values for all blocks
    findBoundingBoxKernel<<<blocks,threads>>>(pc, minX, maxX, minY, maxY, minZ, maxZ); 
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find X extrema in remaining array
    findExtremaKernel<<<1, threads>>>(pc, blocks, minX, maxX, mins, maxes, 0);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find Y extrema in remaining array
    findExtremaKernel<<<1, threads>>>(pc, blocks, minY, maxY, mins, maxes, 1);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find Z extrema
    findExtremaKernel<<<1, threads>>>(pc, blocks, minZ, maxZ, mins, maxes, 2);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Should print out 2,9,0,7,1,6

    //Free memory
    cudaFree(minX);
    cudaFree(maxX);
    cudaFree(minY);
    cudaFree(maxY);
    cudaFree(minZ);
    cudaFree(maxZ);
    std::cerr <<"Find bound complete\n";

}
__global__ void zeroBinsKernel(int* binCount, int partitions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < partitions*partitions*partitions){
        binCount[idx] = 0;
    }
}

/*
This kernel will use a hash function to determine which bin the point hashes into
and will then atomically count the number of points to be added to the bin. 
THERE IS DEFINITELY A BETTER WAY TO DO THIS STEP
*/

__global__ void buildBinsKernel(GPU_Cloud_F4 pc, int* binCount, int partitions, 
                                        float* min, float* max, int** bins, int* memo) {
    
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(ptIdx >= pc.size) return;

    //Copy Global to registry memory
    sl::float4 data = pc.data[ptIdx];

    int binNum = hashToBin(data, min, max, partitions);

    //Find total number of elements in each bin
    int place = atomicAdd(&binCount[binNum],1);
   
    //Make intermediary step to write to global memory. Could avoid this by syncing
    //all blocks
    memo[3*ptIdx] = ptIdx;
    memo[3*ptIdx+1] = binNum;
    memo[3*ptIdx+2] = place;
}

__global__ void mallocBinsKernel(int partitions, int** bins, int* binCount) {
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;

    //Dynamically allocate memory for bins in kernel. Memory must be freed
    //in a different Kernel. It cannot be freed with cudaFree()
    //By definition of the hash function there will be partitions^3 bins 
    if(ptIdx < partitions*partitions*partitions) {
        bins[ptIdx] = (int*)malloc(sizeof(int)*(binCount[ptIdx]));
    }
}

__global__ void assignBinsKernel(int size, int** bins, int* memo) {
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(ptIdx >= size) return;

    //Memory now exists, so write index to global memory
    bins[memo[3*ptIdx+1]][memo[3*ptIdx+2]] = memo[3*ptIdx];

    //printf("(%i, %i, %i), ", memo[3*ptIdx+1], memo[3*ptIdx+2], bins[memo[3*ptIdx+1]][memo[3*ptIdx+2]]);
}

__global__ void freeBinsKernel(int* binCount, int** bins, int partitions){
    
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;

    //If valid bin
    if(ptIdx < partitions*partitions*partitions){
        int* ptr = bins[ptIdx];
        //If memory was allocated
        if(ptr != NULL)
            free(ptr);
    }
}


/*
This function builds the bins that are needed to prevent an O(n^2) search time
for nearest neighbors. Uses min and max values to construct a cube that can be 
divided up into a specified number of partitions on each axis. 
*/
void EuclideanClusterExtractor::buildBins(GPU_Cloud_F4 &pc) {
    int threads = MAX_THREADS;
    int blocks = ceilDiv(pc.size, threads);
    int* memo;
    
    //Allocate memory
    checkStatus(cudaMalloc(&bins, sizeof(int*) * partitions*partitions*partitions));
    checkStatus(cudaMalloc(&binCount, sizeof(int) * partitions*partitions*partitions));
    checkStatus(cudaMalloc(&memo, sizeof(int) * 3 * pc.size));
    
    //Zero Bins
    zeroBinsKernel<<<ceilDiv(partitions*partitions*partitions, threads), threads>>>(binCount, partitions);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Construct the bins to be used
    buildBinsKernel<<<blocks, threads>>>(pc, binCount, partitions, mins, maxes, bins, memo);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Allocates appropriate memory for bins
    //Used because couldn't figure out how to sync blocks
    mallocBinsKernel<<<ceilDiv(partitions*partitions*partitions, threads), threads>>>(partitions, bins, binCount);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Assign values to the created bin structure
    //Used because couldn't figure out how to sync blocks
    assignBinsKernel<<<blocks, threads>>>(pc.size, bins, memo);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Should print something like (5, 0, 0), (6, 0, 1), (3, 0, 2), (7, 0, 3), (0, 0, 4), (3, 1, 5), (1, 0, 6), (2, 0, 7), (6, 1, 8), (1, 1, 9)
    //Don't worry if middle number differs

    //Free memory
    checkStatus(cudaFree(memo));
}

/*
This function frees dynamically allocated memory in buildBins function
*/
void EuclideanClusterExtractor::freeBins() {
    int threads = MAX_THREADS;
    int blocks = ceilDiv(partitions*partitions*partitions, threads);
    
    freeBinsKernel<<<blocks,threads>>>(binCount, bins, partitions);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    checkStatus(cudaFree(binCount));
    checkStatus(cudaFree(bins));
}

__device__ void findEdgePtsOfRadius (sl::float4 &startBinPt, sl::float4 &xBoundPt, sl::float4 &yBoundPt, 
                                    sl::float4 &zBoundPt, int tolerance, float* mins, float* maxes) {
    //Start Bin checking for going out of bounds
    startBinPt.z = (mins[2] < startBinPt.z-tolerance) ? startBinPt.z-tolerance : mins[2]+1;
    startBinPt.y = (mins[1] < startBinPt.y-tolerance) ? startBinPt.y-tolerance : mins[1]+1;
    startBinPt.x = (mins[0] < startBinPt.x-tolerance) ? startBinPt.x-tolerance : mins[0]+1; 
    
    //X Edge checking for going out of bounds
    xBoundPt.z = (mins[2] < xBoundPt.z-tolerance) ? xBoundPt.z-tolerance : mins[2]+1;
    xBoundPt.y = (mins[1] < xBoundPt.y-tolerance) ? xBoundPt.y-tolerance : mins[1]+1;
    xBoundPt.x = (maxes[0] > xBoundPt.x+tolerance) ? xBoundPt.x+tolerance : maxes[0]-1; 
    
    //Y Edge checking for going out of bounds
    yBoundPt.z = (mins[2] < yBoundPt.z-tolerance) ? yBoundPt.z-tolerance : mins[2]+1;
    yBoundPt.y = (maxes[1] > yBoundPt.y+tolerance) ? yBoundPt.y+tolerance : maxes[1]-1;
    yBoundPt.x = (mins[0] < yBoundPt.x-tolerance) ? yBoundPt.x-tolerance : mins[0]+1;                                             
    
    //Z Edge checking for going out of bounds
    zBoundPt.z = (maxes[2] > zBoundPt.z+tolerance) ? zBoundPt.z+tolerance : maxes[2]-1;
    zBoundPt.y = (mins[1] < zBoundPt.y-tolerance) ? zBoundPt.y-tolerance : mins[1]+1;
    zBoundPt.x = (mins[0] < zBoundPt.x-tolerance) ? zBoundPt.x-tolerance : mins[0]+1;                                             
}

/*
This kernel determines the structure of the graph but does not build it
In theory, there is a memory-compute trade off to be made here. This kernel
is not strictly necessary if we allow an upper bound of memory so that each 
point can have the entire dataset amount of neighbors. Perhaps we can 
explore this allocation method instead.
*/
//b: enough, t: each point
__global__ void determineGraphStructureKernel(GPU_Cloud_F4 pc, float tolerance, int* listStart, int** bins, int* binCount,
                                            float* mins, float* maxes, int partitions) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;

    //Initialize edge vars
    sl::float4 startBinPt = pc.data[ptIdx], xBoundPt = startBinPt, 
    yBoundPt = startBinPt, zBoundPt = startBinPt;

    findEdgePtsOfRadius(startBinPt, xBoundPt, yBoundPt, zBoundPt, tolerance, mins, maxes);

    //Find Edge Bins
    int xStartBin = hashToBin(startBinPt, mins, maxes, partitions);
    int xBoundBin = hashToBin(xBoundPt, mins, maxes, partitions);
    int yBoundBin = hashToBin(yBoundPt, mins, maxes, partitions);
    int zBoundBin = hashToBin(zBoundPt, mins, maxes, partitions);

    int yStartBin = xStartBin, zStartBin = xStartBin;

    const int totalBins = (zBoundBin-xStartBin+1) * ((yBoundBin-xStartBin)/partitions+1) *
                    ((xBoundBin-xStartBin)/(partitions*partitions)+1);

    int* binsToSearch = (int*)malloc(sizeof(int)*totalBins);

    /*
    The bin one away from currBin in Z direction is binNum +-1
    The bin one away from currBin in Y direction is binNum +-partitions
    The bin one away from currBin in X direction is binNum +-partitions^2
    */
    //Start at lower left, iterate front to back, bottom to top, left to right 
    //Iterate left to right
    int binAdded = 0;
    int zDif = zBoundBin-zStartBin;
    for(int i = xStartBin; i <= xBoundBin; i += partitions*partitions) {
        //Iterate bottom to top
        for(int j = yStartBin; j <= yBoundBin; j += partitions) {
            //Iterate front to back
            for(int k = zStartBin; k <= zBoundBin; ++k){
                binsToSearch[binAdded] = k;
                binAdded++;
            }
            zBoundBin += partitions; //Shift zBoundBin up
            zStartBin += partitions; //Shift ztartBin up
        }
        yBoundBin += (partitions*partitions); //Shift yBoundBin right
        yStartBin += (partitions*partitions); //Shift yStartBin right
        zStartBin = yStartBin;
        zBoundBin = zStartBin+zDif;
    }

    //Iterate through points in bins to search and check if they are within the radius of the point
    for(size_t i = 0; i < totalBins; ++i){
        for(int j = 0; j < binCount[binsToSearch[i]]; ++j){
            sl::float3 dvec = (pt - sl::float3(pc.data[bins[binsToSearch[i]][j]]));
            
            //this is a neighbor
            if( dvec.norm() < tolerance && bins[binsToSearch[i]][j] != ptIdx) {
                neighborCount++;
            }
        }
    }
    listStart[ptIdx] = neighborCount;
    free(binsToSearch);
    //we must do an exclusive scan using thrust after this kernel
    //printf("%d: %d \n",ptIdx, neighborCount );
}


/* This kernel builds the graph 
Fairly standard adjacency list structure. 
*/
__global__ void buildGraphKernel(GPU_Cloud_F4 pc, float tolerance, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2, 
                            int** bins, int* binCount, float* mins, float* maxes, int partitions) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    //get the adjacency list for this point
    int* list = neighborLists + listStart[ptIdx];
    
     //Initialize edge vars
    sl::float4 startBinPt = pc.data[ptIdx], xBoundPt = startBinPt, 
    yBoundPt = startBinPt, zBoundPt = startBinPt;

    findEdgePtsOfRadius(startBinPt, xBoundPt, yBoundPt, zBoundPt, tolerance, mins, maxes);

    //Find Edge Bins
    int xStartBin = hashToBin(startBinPt, mins, maxes, partitions);
    int xBoundBin = hashToBin(xBoundPt, mins, maxes, partitions);
    int yBoundBin = hashToBin(yBoundPt, mins, maxes, partitions);
    int zBoundBin = hashToBin(zBoundPt, mins, maxes, partitions);

    int yStartBin = xStartBin, zStartBin = xStartBin;

    const int totalBins = (zBoundBin-xStartBin+1) * ((yBoundBin-xStartBin)/partitions+1) *
                    ((xBoundBin-xStartBin)/(partitions*partitions)+1);

    int* binsToSearch = (int*)malloc(sizeof(int)*totalBins);

    /*
    The bin one away from currBin in Z direction is binNum +-1
    The bin one away from currBin in Y direction is binNum +-partitions
    The bin one away from currBin in X direction is binNum +-partitions^2
    */
    //Start at lower left, iterate front to back, bottom to top, left to right 
    //Iterate left to right
    int binAdded = 0;
    int zDif = zBoundBin-zStartBin;
    for(int i = xStartBin; i <= xBoundBin; i += partitions*partitions) {
        //Iterate bottom to top
        for(int j = yStartBin; j <= yBoundBin; j += partitions) {
            //Iterate front to back
            for(int k = zStartBin; k <= zBoundBin; ++k){
                binsToSearch[binAdded] = k;
                binAdded++;
            }
            zBoundBin += partitions; //Shift zBoundBin up
            zStartBin += partitions; //Shift ztartBin up
        }
        yBoundBin += (partitions*partitions); //Shift yBoundBin right
        yStartBin += (partitions*partitions); //Shift yStartBin right
        zStartBin = yStartBin;
        zBoundBin = zStartBin+zDif;
    }

    //Iterate through points in bins to search and check if they are within the radius of the point
    for(size_t i = 0; i < totalBins; ++i){
        for(int j = 0; j < binCount[binsToSearch[i]]; ++j){
            sl::float3 dvec = (pt - sl::float3(pc.data[bins[binsToSearch[i]][j]]));
            
            //this is a neighbor
            if( dvec.norm() < tolerance && bins[binsToSearch[i]][j] != ptIdx) {
                list[neighborCount] = bins[binsToSearch[i]][j]; 
                neighborCount++;
            }
        }
    }

    free(binsToSearch);
    //we must do an exclusive scan using thrust after this kernel
    //printf("%d: %d \n",ptIdx, neighborCount );

    labels[ptIdx] = ptIdx;
    f1[ptIdx] = true;
    f2[ptIdx] = false;
}

__global__ void determineGraphStructureKernelN2(GPU_Cloud_F4 pc, float tolerance, int* listStart) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {
        sl::float3 dvec = (pt - sl::float3(pc.data[i]));
        //this is a neighbor
        if( dvec.norm() < tolerance && i != ptIdx) {
            neighborCount++;
        }
    }
    listStart[ptIdx] = neighborCount;

    //we must do an exclusive scan using thrust after this kernel
    //printf("%d: %d \n",ptIdx, neighborCount );
}


/* This kernel builds the graph 
Fairly standard adjacency list structure. 
*/
__global__ void buildGraphKernelN2(GPU_Cloud_F4 pc, float tolerance, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    //get the adjacency list for this point
    int* list = neighborLists + listStart[ptIdx];
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {

        sl::float3 dvec = (pt - sl::float3(pc.data[i]));
        //this is a neighbor
        if( dvec.norm() < tolerance && i != ptIdx) {
            list[neighborCount] = i;
            neighborCount++;
        }
    }
    
    labels[ptIdx] = ptIdx;
    f1[ptIdx] = true;
    f2[ptIdx] = false;
}

/*
this kernel propogates labels, it must be called in a loop until its flag "m" is false, indicating
no more changes are pending. 
*/
//each thread is a point 
__global__ void propogateLabels(GPU_Cloud_F4 pc, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2, bool* m) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    if(ptIdx == -1){
        for(int i = 0; i < 10; i++){
            printf("Pt %i: ", i);
            for(int j = listStart[i]; j < listStart[i+1]; ++j){
                printf("%i, ", neighborLists[j]);
            }
            printf("\n");    
        }
        
    }
    //debug lines
   // if(threadIdx.x == 0) *m = false;
   // __syncthreads();
   // printf("pt idx: %d, label: %d, flag: %d frontier one: %d frontier two: %d \n", ptIdx, labels[ptIdx], (*m) ? 1 : 0, f1[ptIdx] ? 1 : 0, f2[ptIdx] ? 1 : 0);

    bool mod = false;
    //TODO, load the NEIGHBOR list to shared memory 
    if(f1[ptIdx]) {
        //printf("active frontier %d \n", ptIdx);

        int* list = neighborLists + listStart[ptIdx];
        int listLen = listStart[ptIdx+1] - listStart[ptIdx];
        f1[ptIdx] = false;
        int myLabel = labels[ptIdx];

        //printf("[len] pt idx: %d, list-len: %d \n", ptIdx, listLen);

        for(int i = 0; i < listLen; i++) {
            int otherLabel = labels[list[i]];
            if(myLabel < otherLabel) { //are these reads actually safe?
                //printf("-- updating other: %d to be %d \n", otherLabel, myLabel);

                atomicMin(&labels[list[i]], myLabel);
                f2[list[i]] = true;
                *m = true;
            } else if(myLabel > otherLabel) {
                myLabel = otherLabel;
                mod = true;
            }
        }

        if(mod) {
            atomicMin(&labels[ptIdx], myLabel);
            f2[ptIdx] = true;
            *m = true;
        }
    } 

    /*
    __syncthreads();
    if(threadIdx.x == 0) {
    if(*m) printf("still going \n");
    else printf("done \n");
    }*/
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

//this debug kernel colors points based on their label
__global__ void colorClusters(GPU_Cloud_F4 pc, int* labels, int* keys, int* values, int minCloudSize, int numClusters, float* minX, float* maxX, float* minY, float* maxY, float* minZ, float* maxZ) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    //DEBUG STEP REMOVE
    //pc.data[ptIdx].w = 9.18340948595e-41;
    //return;

    int i = 0;
    while(true) {
        if(labels[ptIdx] == keys[i]) {
            if(values[i] < minCloudSize) {
                pc.data[ptIdx].w = VIEWER_BGR_COLOR;
                return;
            }
            else break;
        }
        i++;
    }
    
    //float red = 3.57331108403e-43;
    //float green = 9.14767637511e-41;
    //float blue = 2.34180515203e-38;
    //float magenta = 2.34184088514e-38; 
    float yellow = 9.18340948595e-41;
    
    pc.data[ptIdx].w = yellow+0.0000000000000001*labels[ptIdx]*4;
    
    //X
    atomicMinFloat(&minX[i], pc.data[ptIdx].x);
    atomicMaxFloat(&maxX[i], pc.data[ptIdx].x);

    //Y
    atomicMinFloat(&minY[i], pc.data[ptIdx].y);
    atomicMaxFloat(&maxY[i], pc.data[ptIdx].y);

    //Z
    atomicMinFloat(&minZ[i], pc.data[ptIdx].z);
    atomicMaxFloat(&maxZ[i], pc.data[ptIdx].z);
}

//this is practically serial, can we just color using OpenGL functions
__global__ void colorExtrema(GPU_Cloud_F4 pc, int* values, int minSize, int* labels, int numClustersOrig, int* validClustersCount, float* minX, float* maxX,  float* minY, float* maxY, float* minZ, float* maxZ) {
    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(clusterIdx >= numClustersOrig) return;

    int place = 0;
    if(values[clusterIdx] > minSize) place = atomicAdd(validClustersCount, 1);
    else return;

  
    pc.data[place*2] = sl::float4(minX[clusterIdx], (minY[clusterIdx] + maxY[clusterIdx])/2, minZ[clusterIdx], 0.0);
    pc.data[place*2+1] = sl::float4(maxX[clusterIdx], (minY[clusterIdx] + maxY[clusterIdx])/2, minZ[clusterIdx], 0.0);
        
    //serailze the extrema into a float4 vector using the "place"
}

__global__ void colorClustersNew(GPU_Cloud_F4 pc, int* labels, int* keys, int numClusters) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    float yellow = 9.18340948595e-41;

    for(int i = 0; i < numClusters; i++) {
        if(labels[ptIdx] == keys[i]) {
            pc.data[ptIdx].w = yellow+0.0000000000000001*labels[ptIdx]*4;
            return;
        }
    }

    pc.data[ptIdx].w = VIEWER_BGR_COLOR;

}

    
class is_smaller_than_min {
public: 
    is_smaller_than_min(int min) : min(min) {}
    __device__ __host__ bool operator()(const int size) {
        return size < min;
    }
private:
    int min;
};

EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize, GPU_Cloud_F4 pc, int partitions) 
: tolerance{tolerance}, minSize{minSize}, maxSize{maxSize}, partitions{partitions} {

    cudaMalloc(&listStart, sizeof(int)*(pc.size+1));
    cudaMalloc(&labels, sizeof(int)*pc.size);
    cudaMalloc(&f1, sizeof(bool)*pc.size);
    cudaMalloc(&f2, sizeof(bool)*pc.size);
    cudaMalloc(&stillGoing, sizeof(bool));

    //Nearest Neighbor Bins
    checkStatus(cudaMalloc(&mins, sizeof(int) * 3));
    checkStatus(cudaMalloc(&maxes, sizeof(int) * 3));

   // colorClusters<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, nullptr);
}


//perhaps use dynamic parallelism 
EuclideanClusterExtractor::ObsReturn EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {
    ObsReturn empty;
    empty.size = 0;
    if(pc.size == 0) return empty;
    //set frontier arrays appropriately [done in build graph]
    //checkStatus(cudaMemsetAsync(f1, 1, sizeof(pc.size)));
    //checkStatus(cudaMemsetAsync(f2, 0, sizeof(pc.size)));
    std::cerr <<"Determining Graph Structure\n";
    //determineGraphStructureKernel<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, listStart, bins, binCount, mins, maxes, partitions);
    determineGraphStructureKernelN2<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, listStart);
    std::cerr <<"Structure Determined\n";
    thrust::exclusive_scan(thrust::device, listStart, listStart+pc.size+1, listStart, 0);
    checkStatus(cudaGetLastError());
    checkStatus(cudaDeviceSynchronize());
    int totalAdjanecyListsSize;
    /*//debugint* temp = (int*) malloc(sizeof(int)*(pc.size+1));
    checkStatus(cudaMemcpy(temp, listStart, sizeof(int)*(pc.size+1), cudaMemcpyDeviceToHost));
    for(int i = 0; i < pc.size+1; i++) std::cout << "ex scan: " << temp[i] << std::endl; */
    checkStatus(cudaMemcpy(&totalAdjanecyListsSize, &listStart[pc.size], sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "total adj size: " << totalAdjanecyListsSize << std::endl;
    std::cerr<<"Building graph kernel\n";
    cudaMalloc(&neighborLists, sizeof(int)*totalAdjanecyListsSize);
    //buildGraphKernel<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, neighborLists, listStart, labels, f1, f2,
      //                                  bins, binCount, mins, maxes, partitions);
    buildGraphKernelN2<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, neighborLists, listStart, labels, f1, f2);
    std::cerr<<"Graph kernel built\n";
    checkStatus(cudaGetLastError());
    checkStatus(cudaDeviceSynchronize());
    

    
    bool stillGoingCPU = true;    
    while(stillGoingCPU) {
        //one iteration of label propogation
        stillGoingCPU = false;
        cudaMemcpy(stillGoing, &stillGoingCPU, sizeof(bool), cudaMemcpyHostToDevice);
        propogateLabels<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, neighborLists, listStart, labels, f1, f2, stillGoing);

        //swap the frontiers
        bool* t = f1;
        f1 = f2;
        f2 = t;

        //get flag to see if we are done
        cudaMemcpy(&stillGoingCPU, stillGoing, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    //Build useful data structures mapping points to clusters and clusters to number of points
    //Let C be the number of clusters, and N the number of points in the cloud
    //After we preform the operations in this block, the contents of the vectors are as follows:
    thrust::device_vector<int> labelsSorted(pc.size); //Point labels sorted by cluster. Len(N). 
    thrust::device_vector<int> count(pc.size, 1); //buffer of all 1s. Len(N)
    thrust::device_vector<int> keys(pc.size); //Each clusters unique ID in ascending order Len(C)
    thrust::device_vector<int> values(pc.size); //The number of points in each cluster in ascending order by ID. Len(C)

    thrust::copy(thrust::device, labels, labels+pc.size, labelsSorted.begin()); //first make the labels sorted contain the labels in order of points
    thrust::sort(thrust::device, labelsSorted.begin(), labelsSorted.end()); //now sort the labels by their label idx, 
    auto pair = thrust::reduce_by_key(thrust::device, labelsSorted.begin(), labelsSorted.end(), count.begin(), keys.begin(), values.begin()); //remove duplicate labels and determine the number of points belonging to each label    
   
    //Determine how many clusters there actually are
    
    int numClustersOrig = thrust::distance(keys.begin(), pair.first);
    std::cout << "CLUSTERS ORIG: " << numClustersOrig << std::endl; 

    float *minX, *maxX, *minY, *maxY, *minZ, *maxZ; 
    cudaMalloc(&minX, sizeof(float)*numClustersOrig);
    cudaMalloc(&maxX, sizeof(float)*numClustersOrig);
    cudaMalloc(&minY, sizeof(float)*numClustersOrig);
    cudaMalloc(&maxY, sizeof(float)*numClustersOrig);
    cudaMalloc(&minZ, sizeof(float)*numClustersOrig);
    cudaMalloc(&maxZ, sizeof(float)*numClustersOrig);
    thrust::fill(thrust::device, minX, minX + numClustersOrig, std::numeric_limits<float>::max());
    thrust::fill(thrust::device, maxX, maxX + numClustersOrig, -std::numeric_limits<float>::max());
    thrust::fill(thrust::device, minY, minY + numClustersOrig, std::numeric_limits<float>::max());
    thrust::fill(thrust::device, maxY, maxY + numClustersOrig, -std::numeric_limits<float>::max());
    thrust::fill(thrust::device, minZ, minZ + numClustersOrig, std::numeric_limits<float>::max());
    thrust::fill(thrust::device, maxZ, maxZ + numClustersOrig, -std::numeric_limits<float>::max());

    /*
    //Now get a list of cluster ID keys that are bigger than the min size by removing those that are less than the min size
    is_smaller_than_min pred(minSize);
    auto keyEnd = thrust::remove_if(thrust::device, keys.begin(), keys.end(), values.begin(), pred);
    thrust::remove_if(thrust::device, values.begin(), values.end(), pred);

    int numClusters = keyEnd - keys.begin();
    keys.resize(numClusters);
    values.resize(numClusters);
    std::cout << "CLUSTERS NEW: " << numClusters << std::endl; */

    //find interest points
    //exculsive scan on values to give the indicies of each new cluster start in the points array 
    //for each on the array returned by the exclusive scan, going from the prev element to the cur,
    //first determine if the labels for that range are contained within the clusterIDs [keys] vector (binary search),
    //if so, then find extrema, otherwise move on

    //Call a kernel to color the clusters for debug reasons
    int* gpuKeys = thrust::raw_pointer_cast( keys.data() );
    int* gpuVals = thrust::raw_pointer_cast( values.data() );
    colorClusters<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, labels, gpuKeys, gpuVals, minSize, numClustersOrig, minX, maxX, minY, maxY, minZ, maxZ);

    int * validClustersCount;
    cudaMalloc(&validClustersCount, sizeof(int));
    cudaMemset(validClustersCount, 0, sizeof(int));
    //colorExtrema<<<ceilDiv(numClustersOrig, MAX_THREADS), MAX_THREADS >>>(pc, gpuVals, minSize, labels, numClustersOrig, validClustersCount, minX, maxX, minY, maxY, minZ, maxZ);

    float *minXCPU, *maxXCPU, *minYCPU, *maxYCPU, *minZCPU, *maxZCPU; 
    minXCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    maxXCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    minYCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    maxYCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    minZCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    maxZCPU = (float*) malloc(sizeof(float)*numClustersOrig);
    cudaMemcpy(minXCPU, minX, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);
    cudaMemcpy(maxXCPU, maxX, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);
    cudaMemcpy(minYCPU, minY, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);
    cudaMemcpy(maxYCPU, maxY, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);
    cudaMemcpy(minZCPU, minZ, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);
    cudaMemcpy(maxZCPU, maxZ, sizeof(float)*numClustersOrig, cudaMemcpyDeviceToHost);

    int* leftBearing;
    int* rightBearing;
    int* leftCPU;
    int* rightCPU; 

    leftCPU = (int*) malloc(sizeof(int));
    rightCPU = (int*) malloc(sizeof(int));

    cudaMalloc(&leftBearing, sizeof(float));
    cudaMalloc(&rightBearing, sizeof(float));
    
    //Laucnh kernels to find clear paths using mins and max cluster arrasy
    findClearPathKernel<<<1, MAX_THREADS>>>(minX, maxX, minZ, maxZ, numClustersOrig, leftBearing, rightBearing);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    findAngleOffCenterKernel<<<1, MAX_THREADS>>>(minX, maxX, minZ, maxZ, numClustersOrig, leftBearing, 0);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    findAngleOffCenterKernel<<<1, MAX_THREADS>>>(minX, maxX, minZ, maxZ, numClustersOrig, rightBearing, 1);    
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    
    //Copy bearings to CPU and display the bearings
    cudaMemcpy(leftCPU, leftBearing, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rightCPU, rightBearing, sizeof(int), cudaMemcpyDeviceToHost);
    
    //Set CPU bearings for viewer use
    bearingRight = *rightCPU;
    bearingLeft = *leftCPU;
    
    cudaFree(leftBearing);
    cudaFree(rightBearing);
    free(leftCPU);
    free(rightCPU);
    
    checkStatus(cudaDeviceSynchronize()); //not needed?
    cudaFree(neighborLists);
    cudaFree(minX);
    cudaFree(maxX);
    cudaFree(minY);
    cudaFree(maxY);
    cudaFree(minZ);
    cudaFree(maxZ);

    int validClustersCPU;
    cudaMemcpy(&validClustersCPU, validClustersCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "valid cluster size: " << validClustersCPU << std::endl;

    ObsReturn obsReturn;
    obsReturn.size = numClustersOrig;
    obsReturn.minX = minXCPU;
    obsReturn.maxX = maxXCPU;
    obsReturn.minY = minYCPU;
    obsReturn.maxY = maxYCPU;
    obsReturn.minZ = minZCPU;
    obsReturn.maxZ = maxZCPU;
    return obsReturn;
}

EuclideanClusterExtractor::~EuclideanClusterExtractor() {
    checkStatus(cudaFree(mins));
    checkStatus(cudaFree(maxes));
}
