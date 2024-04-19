#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "message_filters/subscriber.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"


//TODO: remove hardcoded frames, params;
class TransformedPoints
{
  public:
    TransformedPoints()
    : tf_listener_(tf_buffer_),
      tf2_filter_(sub_, tf_buffer_, "rgb_camera_link", 10, 0)
    {
      sub_.subscribe(nh_, "points2", 10);
      tf2_filter_.registerCallback(&TransformedPoints::points_callback, this);
      pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points2_transformed", 10);
    }

  private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_;
    tf2_ros::MessageFilter<sensor_msgs::PointCloud2> tf2_filter_;
    ros::Publisher pub_;
    sensor_msgs::PointCloud2 transformed_pcd2;

    void points_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
      try
      {
        tf_buffer_.transform<sensor_msgs::PointCloud2>(*msg, transformed_pcd2, "base", ros::Duration(1.));
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("%s", ex.what());
        ros::Duration(1.0).sleep();
        return;
      }
      ROS_INFO("Out callback");
      pub_.publish(transformed_pcd2);
    }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "points2_transformed");
  TransformedPoints tp;
  ros::spin();
}
