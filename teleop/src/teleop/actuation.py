import abc
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface
from robotiq_control.cmodel_urcap import RobotiqCModelURCap


TCPPose = List[float]  # [translation, quaternion]
JointPos = List[float]


class ActuationNode(abc.ABC):
    """Provides arm and gripper actuation."""

    @abc.abstractmethod
    def moveL(self, pose: TCPPose) -> None:
        """Move arm to position in a tool-space."""

    @abc.abstractmethod
    def moveJ(self, position: JointPos) -> None:
        """Move arm to position in a joint-space."""

    @abc.abstractmethod
    def servoL(self, pose: TCPPose) -> None:
        """Servo to position (linear in tool-space)."""

    @abc.abstractmethod
    def servoStop(self) -> None:
        """Stop servo."""

    @abc.abstractmethod
    def gripper_move_and_wait(self, pos: int) -> None:
        """Gripper move *blocking* call. Pos is defined in 0-255 range."""


class SocketActuation(ActuationNode):
    """Low-level RTDE commands and slow gripper."""

    def __init__(self,
                 host: str,
                 rtde_frequency: float = 300.,
                 ) -> None:
        flags = RTDEControl.FLAG_USE_EXT_UR_CAP | RTDEControl.FLAG_UPLOAD_SCRIPT
        self.rtde_c = RTDEControl(
            host,
            frequency=rtde_frequency,
            flags=flags
        )
        self.rtde_r = RTDEReceiveInterface(host, frequency=rtde_frequency)
        self.gripper = RobotiqCModelURCap(host)
        self.gripper.activate(auto_calibrate=False)

    def moveL(self, pose: TCPPose) -> None:
        pose1 = self._convert_rotation(pose)
        pose0 = self.rtde_r.getActualTCPPose()
        z0, z1 = pose0[2], pose1[2]
        if z0 > z1:
            interm = pose1.copy()
            interm[2] = z0
            self.rtde_c.moveL(interm)
        else:
            interm = pose0
            interm[2] = z1
            self.rtde_c.moveL(interm)
        self.rtde_c.moveL(pose1)
        
    def moveJ(self, position: JointPos) -> None:
        self.rtde_c.moveJ(position)
        
    def servoL(self,
               pose: TCPPose,
               time: float = 0.01,
               lookahead_time: float = 0.1,
               gain: float = 100.
               ) -> None:
        pose = self._convert_rotation(pose)
        self.rtde_c.servoL(pose, 0., 0., time, lookahead_time, gain)
        
    def servoStop(self) -> None:
        self.rtde_c.servoStop()
    
    def gripper_move_and_wait(self, pos: int, vel: int = 255, force: int = 255) -> None:
        self.gripper.move_and_wait_for_pos(pos, vel, force)

    @staticmethod
    def _convert_rotation(pose: TCPPose) -> np.ndarray:
        pos, quat = np.split(np.asarray(pose), [3])
        rotvec = Rotation.from_quat(quat).as_rotvec()
        return np.r_[pos, rotvec]
