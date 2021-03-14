#include "common.hpp"

//Find clear path
//Checks to see if minx and maxX are between the two projected lines
//It not checks to see if the line segment between the two minX and maxX intersect the projected line
//If such points are found launch write them to place in shared memory
//If such poitns are not found then in shared memory space write float_max or float_min
//SyncThreads and perform parallel reduction to find minX and maxX thread values
//launch left and right path kernels using new calculated lines
//Left and right path kernels will perform the same steps except in a particular direction
//Assumes we don't have over 1024 clusters

class compareLine {
public:
    int xIntercept;
    float slope;
    
    __device__ compareLine(float angle_in, int xInt_in) : xIntercept{xInt_in}, 
                        slope{tan(angle_in*PI/180)} {
                            if(slope != 0) {
                                slope = 1/slope;
                            }
                        }

    //Returns 1 if point is right of line, 0 if on, -1 if left of line
    __device__ int operator()(float x, float y) {
        
        //Make sure don't divide by 0
        float xc = xIntercept; //x calculated
        if(slope != 0) {
            xc = y/slope+xIntercept; //Find x value on line with same y value as input point
        }
            
        //Point is right of the line
        if(x > xc) {
            return 1;
        }
        //Point is on the line
        else if (x == xc) {
            return 0;
        }
        //Point is left of the line
        else {
            return -1;
        } 
    }

    //Assumes x1 < x2
    __device__ bool operator()(float x1, float y1, float x2, float y2) {
        if(x1 != x2){
            if(slope != 0){
                float slopeSeg = (y2-y1)/(x2-x1);
                float xIntersect = (-slopeSeg*x1+y1-xIntercept)/(slope-slopeSeg);
                return (xIntersect < x2 && xIntersect > x1);
            }
            //Check if left of line and right of line if slope is undefined
            else if(this->operator()(x1,y1) < 1 && this->operator()(x2,y2) > -1) return true; 
        }
        return false;
        
    }
};

int findNextLargestSquare(int num){
    int exp = log2(num) + 1;
    return pow(2,exp);
}

void find_clear_path(float* minX, float* maxX, float* minZ, float* maxZ, int numClusters, int* leftBearing, int* rightBearing);












