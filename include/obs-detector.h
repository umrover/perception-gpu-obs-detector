#include <string>
#include <Camera.h>
#include "recorder.hpp"

/*
 *** Determines where to input clouds for obstacle detection ***
 *      SOURCE_ZED: inputs clouds directly from a connected ZED
 *      SOURCE_GPUMEM: receives a pointer to cloud GPU memory from external source
 *      SOURCE_FILESYSTEM: reads .pc files from specified location
 */
enum DataSource {SOURCE_ZED, SOURCE_GPUMEM, SOURCE_FILESYSTEM}; 

/*
 *** Set up debugging level ***
 */
enum OperationMode {MODE_DEBUG, MODE_SILENT};

/*
 *** Choose which viewer to use ***
 */
enum ViewerType {VIEWER_NONE, VIWER_PCL, VIEWER_GL};

class ObsDetector {
    public:
        //Construct the obstacle detector with appropriate options
        ObsDetector(DataSource source, OperationMode mode, ViewerType viewer);

        //Start recording
        void startRecording(std::string directory);

        //Stop recording
        void stopRecording();

        //Grabs the next frame and performs an obstacle detection
        void update();

    private:
        //Do viewer tick 
        void updateViewer();

        //Sets up detection paramaters from an XML file
        void setupParamaters(std::string parameterFile);


    private: 
        //Viwers

        //Operation paramaters
        DataSource source;
        OperationMode mode;
        ViewerType viewer;
        bool record;

        //Detection algorithms 
        PassThrough passZ;
        RansacPlane ransacPlane;


        //Paramaters
        sl::Resolution cloud_res;
        sl::InitParameters init_params;
        sl::CameraParameters defParams;

        
};