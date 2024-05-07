import copy
import threading
from typing import Any, Dict, NamedTuple

import rospy
import numpy as np
import ros_numpy.point_cloud2 as pc2
import message_filters as mf
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, JointState, PointCloud2
from robotiq_msgs.msg import CModelStatus


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
        # TODO: consider remapping
        self._subs = ROSObservationNode.Observation(
            image=mf.Subscriber('/rgb_to_depth/image_raw', Image),
            depth=mf.Subscriber('/depth/image_rect_raw', Image),
            point_cloud = mf.Subscriber('/points2_transformed', PointCloud2),
            joint_states = mf.Subscriber('joint_states', JointState),
            tcp_frame = mf.Subscriber('tcp_pose', TransformStamped),
            gripper_status=mf.Subscriber('/gripper/status', CModelStatus)
        )
        self._time_filter = mf.ApproximateTimeSynchronizer(self._subs, 15, .1, allow_headerless=False)
        self._time_filter.registerCallback(self._obs_callback)
        # rospy.logerr('%s done init' % rospy.get_name())

    def _obs_callback(self, *args, **kwargs) -> None:
        # rospy.logerr('callback fired: %f', rospy.Time.now().to_sec())
        with self._lock:
            self._obs = ROSObservationNode.Observation(*args, **kwargs)

    def get_observation(self) -> Dict[str, Any]:
        while self._obs is None:
            rospy.loginfo('Waiting for an observation.')
            rospy.sleep(1.)
        with self._lock:
            obs = copy.deepcopy(self._obs)
        xyz, quat = (tr := obs.tcp_frame.transform).translation, tr.rotation
        tcp_pose = np.concatenate([xyz, quat], dtype=np.float32)
        pcd = pc2.pointcloud2_to_array(obs.point_cloud, squeeze=False)
        return {
            'image': self._cvbridge.imgmsg_to_cv2(obs.image, "rgb8"),
            'depth': self._cvbridge.imgmsg_to_cv2(obs.depth),
            'point_cloud': _record_array_to_array(pcd),
            'joint_position': obs.joint_states.position,
            'joint_velocity': obs.joint_states.velocity,
            'tcp_pose': tcp_pose,
            'gripper_pos': obs.gripper_status.gPO / 255.,
            'gripper_is_obj_detected': obs.gripper_status.gOBJ in (1, 2)
        }

def _record_array_to_array(pcd_struct, nan=0., dtype=np.float16):
    # Creates copy which may not be desired.
    assert pcd_struct.ndim == 2, 'HW format is required.'
    pcd = np.zeros(pcd_struct.shape + (3,), dtype=dtype)
    pcd[..., 0] = pcd_struct['x']
    pcd[..., 1] = pcd_struct['y']
    pcd[..., 2] = pcd_struct['z']
    return np.nan_to_num(pcd, copy=False, nan=nan)
