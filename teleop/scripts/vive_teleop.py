#!/usr/bin/env python
import time
import pickle
import pathlib

import rospy
import rospkg
import numpy as np

from ur_env.teleop import vive  # should be replaced with a ROS node

from teleop.scene import Scene


# consider recording rosbag, use ros ur_controllers
class TeleopNode:

    def __init__(self, scene: Scene, fps: float):
        self.scene = scene
        self.spf = 1. / fps
        vr_calibration = rospkg.RosPack().get_path("teleop") + "/etc/vive_calibration.npz"
        self.vr = vive.ViveController(vr_calibration)
        self._displacement = np.zeros(3)

    def _measure_displacement(self) -> np.ndarray:
        vr_state = self.vr.read_state()
        while vr_state.trigger < 1.:
            vr_state = self.vr.read_state()
        tcp_pose = self.scene.get_observation()["tcp_pose"]
        self._displacement = np.asarray(tcp_pose[:3]) - vr_state.position
        while vr_state.trigger > 0.:
            vr_state = self.vr.read_state()
        return self._displacement

    def actuate(self, state: vive.ViveState):
        pos = state.position + self._displacement
        rot = state.rotation.as_quat()
        disc = 1  # 0.5
        grip = np.floor_divide(state.trigger, disc) * disc
        term = state.menu_button
        action = np.r_[pos, rot, grip, term]
        self.scene.actuate(action)

    def collect_demo(self):
        no_state = 0
        obss = []
        cur_t = prev_t = time.time()
        self.scene.initialize_episode()
        self._measure_displacement()
        prev_state = self.vr.read_state()
        while not self.scene.get_termination():
            cur_t = time.time()
            if cur_t - prev_t > self.spf:
                prev_t = cur_t
                obs = self.scene.get_observation()
                obss.append(obs)
            state = self.vr.read_state()
            if not any(state.position):
                self.scene.actuation_node.servoStop()
                raise RuntimeError("Controller is lost.")
            if np.allclose(state.position, prev_state.position, atol=0.07):
                no_state = 0
                prev_state = state
                self.actuate(state)
            else:
                no_state += 1
                if no_state > 10:
                    self.scene.actuation_node.servoStop()
                    rospy.logerr("Discontinuity encountered.")
                    break
        obss.append(self.scene.get_observation())
        return obss


def _read_trackpad(vr: vive.ViveController) -> int:
    val = 0
    while abs(val) < 0.75:
        val = vr.read_state().trackpad[0]
    val = 1 if val > 0 else 0
    print(val)
    return val


def main(dataset_path: str, task: str):
    scene = Scene(tasks=(task,), real_time=True)
    teleop = TeleopNode(scene=scene, fps=8.)
    root = pathlib.Path(dataset_path).resolve()
    task_dir = root / task
    task_dir.mkdir(exist_ok=True)
    while True:
        demo = teleop.collect_demo()
        if any(map(lambda obs: obs["is_terminal"], demo[:-1])) or not demo[-1]["is_terminal"]:
            rospy.logerr('Skipping ill-formed demo')
            continue
        idx = len(list(task_dir.glob("*.pkl"))) + 1  # assumption on contiguous naming.
        print(f"Demo {idx} length: {len(demo)}. Is successful [0/1]?")
        if _read_trackpad(teleop.vr):
            with (task_dir / f"{idx:04d}.pkl").open(mode="wb") as f:
                pickle.dump(demo, f)


if __name__ == "__main__":
    rospy.init_node("teleop")
    try:
        main("/media/robot/Transcend/teleop_dataset", "PutInBox")
    except rospy.ROSInterruptException:
        pass

