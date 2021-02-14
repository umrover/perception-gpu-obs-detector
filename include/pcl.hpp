#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/time.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PointIndices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>


#ifndef PCL
#define PCL

using namespace std;

extern shared_ptr<pcl::visualization::PCLVisualizer> pclViewer; // = pointcloud.createRGBVisualizer();
extern pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_pcl;


void readData();
void setPointCloud(int i);
void pclToZed(sl::Mat &zed, pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pcl);

shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pt_cloud_ptr);


void pclToZed(sl::Mat &zed, pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pcl);
void ZedToPcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & p_pcl_point_cloud, sl::Mat zed_cloud);

#endif