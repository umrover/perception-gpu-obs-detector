
#include <sl/Camera.hpp>
#include "GLViewer.hpp"
//#include "test-filter.hpp"
#include "plane-ransac.hpp"
#include <iostream>
#include "common.hpp"
#include <Eigen/Dense>
#include <chrono> 
#include "test-filter.hpp"
#include "pcl.hpp"
#include<unistd.h>
#include <thread>
#include "pass-through.hpp"
#include "euclidean-cluster.hpp"
#include "driver.hpp"

#include "recorder.hpp"

using namespace std::chrono; 

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/

#define USE_PCL
#define DEBUG 

//Zed camera and viewer
sl::Camera zed;
GLViewer viewer;
int guiK = 0;

RansacPlane::Plane planePoints;
EuclideanClusterExtractor::ObsReturn obstacles;

//This is a thread to just spin the zed viewer
void spinZedViewer() {
    while(viewer.isAvailable()) {
        std::this_thread::sleep_for (std::chrono::milliseconds(10));
     //   updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 600.5);
     
    /*   float minX[] = {0.0};
        float maxX[] = {100.0};
        float minY[] = {0.0};
        float maxY[] = {100.0};
        float minZ[] = {0.0};
        float maxZ[] = {100.0}; */
        updateObjectBoxes(obstacles.size, obstacles.minX, obstacles.maxX, obstacles.minY, obstacles.maxY, obstacles.minZ, obstacles.maxZ );
      // updateObjectBoxes(1, minX, maxX, minY, maxY, minZ, maxZ );
    }
}

void nextFrame() {
    guiK++;
    cout << guiK << endl;
}
void prevFrame() {
    guiK--;
    cout << guiK << endl;
}


int main(int argc, char** argv) {  
    
    sl::Resolution cloud_res(320/2, 180/2);
    int k = 0;
    
    //Setup camera and viewer
    sl::InitParameters init_params;
    init_params.coordinate_units = sl::UNIT::MILLIMETER;
    init_params.camera_resolution = sl::RESOLUTION::VGA; 
    init_params.camera_fps = 100;
    #ifndef USE_PCL
    zed.open(init_params); 
    // init viewer
    auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);
    #endif
    #ifdef USE_PCL
    //sl::CameraInformation camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    //auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    //zed.open(init_params); 
    //auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    //auto &blah = camera_config.calibration_parameters.left_cam;
    //cout << blah.fx << " " << blah.fy << " " << blah.cx << " " << blah.cy << " " << blah.image_size.width << " " << blah.image_size.height << endl;

    sl::CameraParameters  defParams;
    defParams.fx = 79.8502;
    defParams.fy = 80.275;
    defParams.cx = 78.8623;
    defParams.cy = 43.6901;
    defParams.image_size.width = 160;
    defParams.image_size.height = 90;

    GLenum errgl = viewer.init(argc, argv, defParams);
   // return 1;
    #endif

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    int pcSize = cloud_res.area(); 
    cout << "Point clouds are of size: " << pcSize << endl;

    //Pass Through Filter
    PassThrough passZ('z', 200.0, 7000.0); //This probly won't do much since range so large 200-7000
   // PassThrough passY('y', 100.0, 600.0); 
    
    //This is a RANSAC model that we will use
    //sl::float3 axis, float epsilon, int iterations, float threshold,  int pcSize
    RansacPlane ransac(sl::float3(0, 1, 0), 7, 400, 150, pcSize); //change the threshold to 150
        
    //PCL integration variables
    int iter = 0;
	readData(); //Load the pcd file names into pcd_names
	setPointCloud(5); //Set the first point cloud to be the first of the files
    pclViewer = createRGBVisualizer(pc_pcl);

    #ifdef USE_PCL
    //slows down performance of all GPU functions since running in parallel with them
    thread zedViewerThread(spinZedViewer);
    #endif

    GPU_Cloud_F4 tmp;
    tmp.size = cloud_res.width*cloud_res.height;
    EuclideanClusterExtractor ece(100, 50, 0, tmp, 9); //60/120
    
    Recorder blah("blah");
    Reader bleh2("blah");

    while(true) {
        //Todo, Timer class. Timer.start(), Timer.record() 
        k++;

        //Grab cloud from PCD filesl::Camera::CameraParamaters
        #ifdef USE_PCL 
       // setPointCloud( 61+ guiK /*61*/ );
        sl::Mat pclTest(sl::Resolution(320/2, 180/2), sl::MAT_TYPE::F32_C4, sl::MEM::CPU);
       // pclToZed(pclTest, pc_pcl);
       // GPU_Cloud_F4 pc_f4 = getRawCloud(pclTest, true);
        bleh2.load(61 + guiK, pclTest);
        GPU_Cloud_F4 pc_f4 = getRawCloud(pclTest, true);
        //continue;
        //DEBUG STEP, safe to remove if causing slowness - ash
        sl::Mat orig; 
        pclTest.copyTo(orig, sl::COPY_TYPE::GPU_GPU);
        #endif

        //Grab cloud from the Zed camera
        #ifndef USE_PCL        
        auto grabStart = high_resolution_clock::now();
        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        GPU_Cloud pc = getRawCloud(gpu_cloud);
        GPU_Cloud_F4 pc_f4 = getRawCloud(gpu_cloud, true);
        auto grabEnd = high_resolution_clock::now();
        auto grabDuration = duration_cast<microseconds>(grabEnd - grabStart); 
        cout << "grab time: " << (grabDuration.count()/1.0e3) << " ms" << endl; 
        //DEBUG STEP, safe to remove if causing slowness - ash
        //sl::Mat orig; 
        //gpu_cloud.copyTo(orig, sl::COPY_TYPE::GPU_GPU);
        #endif
        blah.writeFrame(gpu_cloud);
        

        //Run PassThrough Filter
        cout << "[size] pre-pass-thru: " << pc_f4.size << endl;
        auto passThroughStart = high_resolution_clock::now();
        passZ.run(pc_f4);
        auto passThroughStop = high_resolution_clock::now();
        auto passThroughDuration = duration_cast<microseconds>(passThroughStop - passThroughStart); 
        cout << "pass-through time: " << (passThroughDuration.count()/1.0e3) << " ms" <<  endl; 

        //DEBUG STEP, safe to remove if causing slowness - ash
        //sl::Mat orig; 
        //pclTest.copyTo(orig, sl::COPY_TYPE::GPU_GPU);
       
        cout << "[size] pre-ransac: " << pc_f4.size << endl; 
        //Perform RANSAC Plane segmentation to find the ground
        auto ransacStart = high_resolution_clock::now();
        planePoints = ransac.computeModel(pc_f4, true);
        auto ransacStop = high_resolution_clock::now();
        auto ransacDuration = duration_cast<microseconds>(ransacStop - ransacStart); 
        cout << "ransac time: " << (ransacDuration.count()/1.0e3) << " ms" <<  endl; 
        cout << "[size] post-ransac: " << pc_f4.size << endl; 
        clearStale(pc_f4, 320/2*180/2);

        
        
        ece.findBoundingBox(pc_f4);
        ece.buildBins(pc_f4);
        auto eceStart = high_resolution_clock::now();
        obstacles = ece.extractClusters(pc_f4);
        auto eceStop = high_resolution_clock::now();
        ece.freeBins();
        
        auto eceDuration = duration_cast<microseconds>(eceStop - eceStart); 
        cout << "ECE time: " << (eceDuration.count()/1.0e3) << " ms" <<  endl; 
        

        #ifndef USE_PCL
        viewer.isAvailable();
        #endif
        
        //PCL viewer + Zed SDK Viewer
        #ifdef USE_PCL
        ZedToPcl(pc_pcl, pclTest);
        pclViewer->updatePointCloud(pc_pcl); //update the viewer 
    	pclViewer->spinOnce(10);
        //viewer.updatePointCloud(pclTest);

        viewer.updatePointCloud(orig);


        #endif


        //ZED sdk custom viewer ONLY
        #ifndef USE_PCL
        //draw an actual plane on the viewer where the ground is
        updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 600.5);
        viewer.updatePointCloud(gpu_cloud);
        updateObjectBoxes(obstacles.size, obstacles.minX, obstacles.maxX, obstacles.minY, obstacles.maxY, obstacles.minZ, obstacles.maxZ );
        std::cerr << "Running update projected Lines\n";
        updateProjectedLines(ece.bearingRight, ece.bearingLeft);
        // updateObjectBoxes(1, minX, maxX, minY, maxY, minZ, maxZ );
       // viewer.updatePointCloud(testcloudmat);

        #endif

        cerr << "Camera frame rate: " << zed.getCurrentFPS() << "\n";
        
        //I did this that way the viewer would still respond
        /*
        for(int i = 0; i < 1000; i++){
            viewer.isAvailable();
        }
        */
        
        //std::this_thread::sleep_for(0.2s);
    }
    gpu_cloud.free();
    zed.close(); 
    return 1;
}
