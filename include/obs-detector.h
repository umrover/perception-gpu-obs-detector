#include <string>
#include <sl/Camera.hpp>
#include "recorder.hpp"
#include "plane-ransac.hpp"
#include "pass-through.hpp"
#include "GLViewer.hpp"
#include "euclidean-cluster.hpp"
#include <thread>
#include "Timer.hpp"
#include "common.hpp"

/*
 *** Determines where to input clouds for obstacle detection ***
 *      SOURCE_ZED: inputs clouds directly from a connected ZED
 *      SOURCE_GPUMEM: receives a pointer to cloud GPU memory from external source
 *      SOURCE_FILESYSTEM: reads .pc files from specified location
 */
enum class DataSource {ZED, GPUMEM, FILESYSTEM}; 

/*
 *** Set up debugging level ***
 */
enum class OperationMode {DEBUG, SILENT};

/*
 *** Choose which viewer to use ***
 */
enum ViewerType {NONE, PCL, GL};

class ObsDetector {
    public:
        //Construct the obstacle detector with appropriate options
        ObsDetector(DataSource source, OperationMode mode, ViewerType viewer);

        //Destructor 
        ~ObsDetector();

        //Start recording
        void startRecording(std::string directory);

        //Stop recording
        void stopRecording();

        //Grabs the next frame and performs an obstacle detection
        void update();

        //Do viewer tick
        void spinViewer();


    private:
        //Do viewer tick 
        //void spinViewer();

        //Sets up detection paramaters from an XML file
        void setupParamaters(std::string parameterFile);


    private: 
        //Data sources
        sl::Camera zed;
        Reader fileReader;

        //Viwers
        std::thread graphicsThread;
        GLViewer glViewer;

        //Operation paramaters
        DataSource source;
        OperationMode mode;
        ViewerType viewer;
        bool record;

        //Detection algorithms 
        PassThrough *passZ;
        RansacPlane *ransacPlane;
        EuclideanClusterExtractor *ece;

        //Paramaters
        sl::Resolution cloud_res;
        sl::InitParameters init_params;
        sl::CameraParameters defParams;
        std::string readDir;
        
        //Output data
        RansacPlane::Plane planePoints;
        EuclideanClusterExtractor::ObsReturn obstacles;

        //Other
        int frameNum = 0;
        
};