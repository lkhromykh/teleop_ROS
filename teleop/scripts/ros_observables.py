#!/usr/bin/env python
import copy
import threading
from typing import Dict, NamedTuple

import rospy
import numpy as np
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
import message_filters as mf
import tf_conversions
import tf2_ros
from tf2_sensor_msgs import do_transform_cloud
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, JointState, PointCloud2
from robotiq_msgs.msg import CModelStatus


Array = np.ndarray


class ROSObservationNode:

    class Observation(NamedTuple):

        image: Image
        depth: Image
        point_cloud: PointCloud2
        joint_states: JointState
        tcp_frame: TransformStamped
        gripper_status: CModelStatus

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cvbridge = CvBridge()
        self._obs = None
        self._subs = ROSObservationNode.Observation(
            image=mf.Subscriber('/rgb/image_raw', Image),
            depth=mf.Subscriber('/depth/image_raw', Image),
            point_cloud = mf.Subscriber('points2', PointCloud2),
            joint_states = mf.Subscriber('joint_states', JointState),
            tcp_frame = mf.Subscriber('tcp_pose', TransformStamped),
            gripper_status=mf.Subscriber('/gripper/status', CModelStatus)
        )
        self._time_filter = mf.ApproximateTimeSynchronizer(self._subs, 10, 10)
        self._time_filter.registerCallback(self._obs_callback)
        self._transform = tf2_ros.Buffer().lookup_transform('base', 'rgb_camera_link', rospy.Time())
        rospy.info('Got transform: %s', self._transform)

    def _obs_callback(self, *args, **kwargs) -> None:
        with self._lock:
            rospy.loginfo('callback fired %d', rospy.Time.now())
            self._obs = ROSObservationNode.Observation(*args, **kwargs)

    def get_observation(self) -> Dict[str, 'Any']:
        with self._lock:
            obs = copy.deepcopy(self._obs)
        pcd = do_transform_cloud(obs.point_cloud, self._transform)
        pcd = pointcloud2_to_xyz_array(pcd, remove_nans=False)
        p, q = (tr := obs.tcp_frame.transform).translation, tr.rotation
        return {
            'image': self._cvbridge.imgmsg_to_cv2(obs.image),
            'depth': self._cvbridge.imgmsg_to_cv2(obs.depth),
            'point_cloud': np.nan_to_num(pcd, copy=False),
            'joint_position': obs.joint_states.position,
            'joint_velocity': obs.joint_states.velocity,
            'tcp_pose': np.float32([p.x, p.y, p.z, q.x, q.y, q.z, q.w]),
            'gripper_pos': obs.gripper_status.gPO / 255.,
            'gripper_is_obj_detected': obs.gripper_status.gOBJ in (2, 3)
        }


if __name__ == '__main__':
    rospy.init_node('teleop')
    obs_node = ROSObservationNode()
    def on_shut():
        obs = obs_node.get_observation()
        import pdb; pdb.set_trace()
    rospy.on_shutdown(on_shut)
    rospy.spin()
