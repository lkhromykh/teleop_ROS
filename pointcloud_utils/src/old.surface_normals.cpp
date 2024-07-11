#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/features/normal_3d_omp.h"
#include "pcl/features/integral_image_normal.h"
//#include "pcl/visualization/pcl_vizualizer.h"


class SurfaceNormals final {
  public:
    SurfaceNormals()
    : in_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      out_cloud_(new pcl::PointCloud<pcl::Normal>)
    {
      sub_ = nh_.subscribe("points2", 2, &SurfaceNormals::callback, this);
      pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points2_wnormals", 2);
    }

  private:
    void callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
      ROS_INFO("In callback %f", ros::Time::now().toSec());
      pcl::fromROSMsg(*msg, *in_cloud_);
      /*
      pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
      ne.setInputCloud(in_cloud_);
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
      ne.setSearchMethod(tree);
      ne.setRadiusSearch(0.03);
      ne.compute(*out_cloud_);
      */
      pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
      ne.setMaxDepthChangeFactor(0.02f);
      ne.setNormalSmoothingSize(10.0f);
      ne.setInputCloud(in_cloud_);
      ne.compute(*out_cloud_);

      pcl::toROSMsg(*out_cloud_, out_msg_);
      ROS_INFO("Publishing %f", ros::Time::now().toSec());
      pub_.publish(out_msg_);

    }

  private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr out_cloud_;
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
