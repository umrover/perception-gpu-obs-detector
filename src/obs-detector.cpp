#include "obs-detector.h"
using namespace std;

ObsDetector::ObsDetector(DataSource source, OperationMode mode, ViewerType viewer) : source(source), mode(mode), viewer(viewer), record(false)
{
    setupParamaters("");
    
    //Init data stream from source
    if(source == DataSource::ZED) {
        zed.open(init_params); 
        auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
        defParams = camera_config.calibration_parameters.left_cam;
    } else if(source == DataSource::FILESYSTEM) {
        fileReader.open(readDir);
    }

    //Init Viewers
    if(mode != OperationMode::SILENT && viewer == ViewerType::PCLV) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
        pclViewer = createRGBVisualizer(pc_pcl);
    } else if(mode != OperationMode::SILENT && viewer == ViewerType::GL) {
        int argc = 1;
        char *argv[1] = {(char*)"Window"};
        glViewer.init(argc, argv, defParams);
    }
};

//TODO: Make it read params from a file
void ObsDetector::setupParamaters(std::string parameterFile) {
    //Operating resolution
    cloud_res = sl::Resolution(320/2, 180/2);
    readDir = "test-record/";

    //Zed params
    init_params.coordinate_units = sl::UNIT::MILLIMETER;
    init_params.camera_resolution = sl::RESOLUTION::VGA; 
    init_params.camera_fps = 100;
    
    //Set the viewer paramas
    defParams.fx = 79.8502;
    defParams.fy = 80.275;
    defParams.cx = 78.8623;
    defParams.cy = 43.6901;
    defParams.image_size.width = 160;
    defParams.image_size.height = 90;

    //Obs Detecting Algorithm Params
    passZ = new PassThrough('z', 200.0, 7000.0); //7000
    ransacPlane = new RansacPlane(sl::float3(0, 1, 0), 7, 400, 150, cloud_res.area());
    ece = new EuclideanClusterExtractor(100, 50, 0, cloud_res.area(), 9); 
}



void ObsDetector::update() {
    if(source == DataSource::ZED) {
        sl::Mat frame; // (cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
        zed.grab();
        zed.retrieveMeasure(frame, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        update(frame);
    } else if(source == DataSource::FILESYSTEM) {
        //DEBUG 
        //frameNum = 250;
        sl::Mat frame(cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::CPU);
        fileReader.load(frameNum, frame);
        update(frame);
    } 
} 

// Call this directly with ZED GPU Memory
void ObsDetector::update(sl::Mat &frame) {
    // Get a copy if debug is enabled
    sl::Mat orig; 
    if(mode != OperationMode::SILENT) {
        frame.copyTo(orig, sl::COPY_TYPE::GPU_GPU);
    }

    // Convert ZED format into CUDA compatible 
    GPU_Cloud_F4 pc; 
    pc = getRawCloud(frame);

    // Processing 
    /*
    passZ->run(pc);
    ransacPlane->computeModel(pc, true);
    obstacles = ece->extractClusters(pc); */

    // Rendering
    if(mode != OperationMode::SILENT) {
        clearStale(pc, cloud_res.area());
        if(viewer == ViewerType::GL) {
            glViewer.updatePointCloud(orig);
        } else {
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
           ZedToPcl(pc_pcl, orig);
           pclViewer->updatePointCloud(pc_pcl); //update the viewer 
        }
    }

    // Recording
    if(record) {
        recorder.writeFrame(orig);
    }

    frameNum++;
}

void ObsDetector::spinViewer() {
    if(viewer == ViewerType::GL) {
        glViewer.isAvailable();
        updateObjectBoxes(obstacles.size, obstacles.minX, obstacles.maxX, obstacles.minY, obstacles.maxY, obstacles.minZ, obstacles.maxZ );
        updateProjectedLines(ece->bearingRight, ece->bearingLeft);
    } else if(viewer == ViewerType::PCLV) {
        pclViewer->spinOnce(10);
    }
}

void ObsDetector::startRecording(std::string directory) {
    recorder.open(directory);
    record = true;
}

 ObsDetector::~ObsDetector() {
     delete passZ;
     delete ransacPlane;
     delete ece;
 }

int main() {
    ObsDetector obs(DataSource::ZED, OperationMode::DEBUG, ViewerType::GL);
    //obs.startRecording("test-record2");
    while(true) {
        obs.spinViewer();
        obs.update();
    }

    return 0;
}