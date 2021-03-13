#include "common.hpp"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#pragma once

/** 
 * \class voxelGrid
 * \brief class that generates Voxel Grid of point cloud
 */
class VoxelGrid {
public: 
    
    /**
     * \brief VoxelGrid constructor
     * \param partitions: number of divisions on each axis of voxel grid
     */
    VoxelGrid(int partitions);

    /**
     * \brief given a PC finds creates the structure of the voxel grid
     * \param pc: GPU point cloud
     * \return void
     */
    void buildBins(GPU_Cloud_F4 &pc);

private:

    int partitions;

};

