import copy
import threading
from typing import Any, Dict, NamedTuple

import rospy
import numpy as np
import ros_numpy.point_cloud2 as pc2
import message_filters as mf
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, JointState, PointCloud2, CameraInfo
from robotiq_msgs.msg import CModelStatus


class ROSObservationNode:

    class Observation(NamedTuple):
        image: Image
        depth: Image
        #point_cloud: PointCloud2
        camera_info: CameraInfo
        joint_states: JointState
        tcp_frame: TransformStamped
        optical_frame: TransformStamped
        gripper_status: CModelStatus

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cvbridge = CvBridge()
        self._obs = None
        self._subs = ROSObservationNode.Observation(
            image=mf.Subscriber("image", Image),
            depth=mf.Subscriber("depth", Image),
            #point_cloud=mf.Subscriber("point_cloud", PointCloud2),
            camera_info=mf.Subscriber("camera_info", CameraInfo),
            joint_states=mf.Subscriber("/joint_states", JointState),
            tcp_frame=mf.Subscriber("/tcp_pose", TransformStamped),
            optical_frame=mf.Subscriber("optical_frame", TransformStamped),
            gripper_status=mf.Subscriber("/gripper/status", CModelStatus)
        )
        # todo: this still cause problems
        self._time_filter = mf.ApproximateTimeSynchronizer(self._subs, 1, .1, allow_headerless=False)
        self._time_filter.registerCallback(self._obs_callback)

    def _obs_callback(self, *args, **kwargs) -> None:
        with self._lock:
            self._obs = ROSObservationNode.Observation(*args, **kwargs)

    def get_observation(self) -> Dict[str, Any]:
        while self._obs is None:
            rospy.loginfo("Waiting for an observation.")
            rospy.sleep(2.)
        with self._lock:
            obs = copy.deepcopy(self._obs)

        def transform_to_pos(tr):
          p, q = tr.translation, tr.rotation
          return p.x, p.y, p.z, q.x, q.y, q.z, q.w
        tcp_pose = transform_to_pos(obs.tcp_frame.transform)
        optical_frame = transform_to_pos(obs.optical_frame.transform)
        #pcd = pc2.pointcloud2_to_array(obs.point_cloud, squeeze=False)
        camera_matrix = np.asarray(obs.camera_info.K).reshape(3,3)
        obs = {
            "image": self._cvbridge.imgmsg_to_cv2(obs.image, "rgb8"),
            "depth": self._cvbridge.imgmsg_to_cv2(obs.depth),
            #"point_cloud": _record_array_to_array(pcd),
            "camera_matrix": camera_matrix,
            "joint_position": obs.joint_states.position,
            "joint_velocity": obs.joint_states.velocity,
            "tcp_pose": tcp_pose,
            "optical_frame": optical_frame,
            "gripper_pos": obs.gripper_status.gPO / 255.,
            "gripper_is_obj_detected": obs.gripper_status.gOBJ in (1, 2)
        }
        return {k: np.asarray(v) for k, v in obs.items()}

def _record_array_to_array(pcd_struct, nan=0., dtype=np.float16):
    # Creates copy which may not be desired.
    assert pcd_struct.ndim == 2, "HW format is required."
    pcd = np.zeros(pcd_struct.shape + (3,), dtype=dtype)
    pcd[..., 0] = pcd_struct["x"]
    pcd[..., 1] = pcd_struct["y"]
    pcd[..., 2] = pcd_struct["z"]
    return np.nan_to_num(pcd, copy=False, nan=nan)

