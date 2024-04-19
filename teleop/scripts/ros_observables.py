#!/usr/bin/env python
import copy
import threading
from typing import Dict, NamedTuple, Tuple

import numpy as np
import rospy
import tf_conversions
import tf2_ros
from cv_bridge import CvBridge
import message_filters as mf
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, JointState, PointCloud2
from robotiq_msgs.msg import CModelStatus
import sensor_msgs.point_cloud2 as pc2


Array = np.ndarray


class ROSObservationNode:

    class Observation(NamedTuple):

        images: Tuple[Image]
        depths: Tuple[Image]
        point_clouds: Tuple[PointCloud2]
        joint_state: JointState
        tcp_frame: TransformStamped
        gripper_status: CModelStatus

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cvbridge = CvBridge()
        self._obs = None
        image_sub = mf.Subscriber('/rgb/image_raw', Image)
        depth_sub = mf.Subscriber('/depth/image_raw', Image)
        points_sub = mf.Subscriber('points2', PointCloud2)
        joint_states_sub = mf.Subscriber('joint_states', JointState)
        tcp_sub = mf.Subscriber('tcp_pose', TransformStamped)
        gripper_sub = mf.Subscriber('/gripper/status', CModelStatus)
        self._subs = [image_sub, depth_sub, points_sub, joint_states_sub, tcp_sub, gripper_sub]
        self._time_filter = mf.ApproximateTimeSynchronizer(self._subs, 10, 10)
        self._time_filter.registerCallback(self._obs_callback)

    def _obs_callback(self, image, depth, pcd, joint_state, tcp_frame, gripper_status) -> None:
        with self._lock:
            self._obs = ROSObservationNode.Observation(
                images=(image,),
                depths=(depth,),
                point_clouds=(pcd,),
                joint_state=joint_state,
                tcp_frame=tcp_frame,
                gripper_status=gripper_status
            )

    def get_observation(self) -> Dict[str, 'Any']:
        while self._obs is None:
            rospy.loginfo('Waiting for an observation')
            rospy.sleep(1.)
        with self._lock:
            obs = copy.deepcopy(self._obs._replace())
        def msgs_to_tuple(fn, msgs): return tuple(map(fn, msgs))
        p, q = (tr := obs.tcp_frame.transform).translation, tr.rotation
        return {
            'images': msgs_to_tuple(self._cvbridge.imgmsg_to_cv2, obs.images),
            'depths': msgs_to_tuple(self._cvbridge.imgmsg_to_cv2, obs.depths),
            'point_clouds': msgs_to_tuple(pc2.read_points_list, obs.point_clouds),
            'joint_position': obs.joint_state.position,
            'joint_velocity': obs.joint_state.velocity,
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
