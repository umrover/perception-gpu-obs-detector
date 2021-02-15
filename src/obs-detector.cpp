#include "obs-detector.h"

ObsDetector::ObsDetector(DataSource source, OperationMode mode, ViewerType viewer) : source(source), mode(mode), viewer(viewer), record(false)
{
    setupParamaters("");

    if(source == SOURCE_ZED) {
        zed.open(init_params); 
        auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
        defParams = camera_config.calibration_parameters.left_cam;
    } else if(source == SOURCE_FILESYSTEM) {

    }

    if(viewer == VIEWER_PCL) {
        //readData(); 
        //setPointCloud(0); 
        //pclViewer = createRGBVisualizer(pc_pcl);

    } else if(viewer == VIEWER_GL) {
        glViewer.init(0, nullptr, defParams);
    }
};

//TODO: Make it read params from a file
void ObsDetector::setupParamaters(std::string parameterFile) {
    cloud_res = sl::Resolution(320/2, 180/2);

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

    //Algos
    passZ = PassThrough('z', 200.0, 7000.0);
    ransacPlane = RansacPlane(sl::float3(0, 1, 0), 7, 400, 150, cloud_res.area()); 
}

void ObsDetector::update() {

}

int main() {
    return 0;
}