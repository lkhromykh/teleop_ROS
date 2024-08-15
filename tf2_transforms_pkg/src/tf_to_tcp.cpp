#include <string>
#include "ros/ros.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/TransformStamped.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tf_to_tcp");
  ros::NodeHandle node("~");
  std::string frame;
  while (!node.getParam("frame", frame))
  {
    ROS_ERROR("Can't fetch frame param.");
    ros::Duration(1.0).sleep();
  }
  ROS_WARN("Got frame name: %s", frame.c_str());

  ros::Publisher pub = node.advertise<geometry_msgs::TransformStamped>("transformed_frame", 10);
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  ros::Rate rate(100.0);
  geometry_msgs::TransformStamped transform;
  while (node.ok())
  {
    try 
    {
      transform = tfBuffer.lookupTransform("base", frame, ros::Time(0));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("%s", ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }
    pub.publish(transform);
    rate.sleep();
  }
  return 0;
}
