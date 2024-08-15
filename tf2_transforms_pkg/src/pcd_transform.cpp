#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/transform_listener.h"
#include "pcl_ros/transforms.h"

constexpr int QUEUE_SIZE = 2;

class TransformedPoints
{
  public:
    TransformedPoints() : tf_listener_(tf_buffer_)
    {
      sub_ = nh_.subscribe("points2", QUEUE_SIZE, &TransformedPoints::points_callback, this);
      pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points2_transformed", QUEUE_SIZE);
    }

  private:
    void points_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
      if (pcl_ros::transformPointCloud("base", *msg, transformed_pcd2, tf_buffer_))
        pub_.publish(transformed_pcd2);
    }

  private:
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    sensor_msgs::PointCloud2 transformed_pcd2;
};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "points2_transformed");
  TransformedPoints tp;
  ros::spin();
}
