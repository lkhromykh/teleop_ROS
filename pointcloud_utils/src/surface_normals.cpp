#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/features/normal_3d_omp.h"
#include "pcl/features/integral_image_normal.h"
#include "pcl/visualization/pcl_visualizer.h"


class SurfaceNormals final {
  public:
    SurfaceNormals()
    : in_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>),
      out_cloud_(new pcl::PointCloud<pcl::Normal>),
      viewer_("PCL Viewer")
    {
      sub_ = nh_.subscribe("points2_transformed", 2, &SurfaceNormals::callback, this);
      pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points2_normals", 2);

      ne_.setNormalEstimationMethod(ne_.AVERAGE_3D_GRADIENT);
      ne_.setMaxDepthChangeFactor(0.02f);
      ne_.setNormalSmoothingSize(10.0f);

      viewer_.setBackgroundColor(0.0, 0.0, 0.0);
      viewer_.addCoordinateSystem(0.0);
      viewer_.initCameraParameters();
      viewer_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    }

  private:
    void callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
      ROS_INFO("In callback %f", ros::Time::now().toSec());
      pcl::fromROSMsg(*msg, *in_cloud_);
      ne_.setInputCloud(in_cloud_);
      ne_.compute(*out_cloud_);
      pcl::toROSMsg(*out_cloud_, out_msg_);
      ROS_INFO("Publishing %f", ros::Time::now().toSec());
      pub_.publish(out_msg_);

#if 1
      viewer_.removeAllPointClouds();
      viewer_.addPointCloud<pcl::PointXYZRGB>(in_cloud_, "cloud");
      viewer_.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(in_cloud_, out_cloud_, 40, 0.03, "normals");
      viewer_.spinOnce();
#endif
    }

  private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr out_cloud_;
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_;
    pcl::visualization::PCLVisualizer viewer_;
    sensor_msgs::PointCloud2 out_msg_;
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
};


int main(int argc, char** argv) {
  ros::init(argc, argv, "surface_normals");
  SurfaceNormals sn;
  ros::spin();
}
