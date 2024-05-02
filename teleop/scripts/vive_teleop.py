#!/usr/bin/env python

import time
import pickle
import logging
import pathlib
import itertools

import rospy
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl

from ur_env.teleop import vive  # should be replaced with a ROS node

from robotiq_control.cmodel_urcap import RobotiqCModelURCap
from teleop.observables import ROSObservationNode


Array = np.ndarray

BALLS = [f'{c} ball' for c in ('red', 'green', 'blue', 'yellow')]
TOYS = ['red santa', 'grey seal', 'pink cat', 'white cat']
CUBES = [f'{c} cube' for c in ('red', 'green', 'blue', 'yellow')]
TASKS = {
    #'PickUp': [f'Pick up the {item}' for item in TOYS],
    #'PutInBox': [f'Put the {item} in the box' for item in TOYS],
    'StackCubes': [f'Place the {c1} on top of the {c2}' for c1, c2 in itertools.combinations(CUBES, 2)],
    #'StackBlocks' : [f'Stack {i} cubes' for i in range(2, 5)],
    #'SlideCubes': [f'Slide the {c1} to the {c2}' for c1, c2 in itertools.combinations(CUBES, 2)],
    #'OpenDrawer': [f'Open the {shelf} drawer' for shelf in ('top', 'middle')]#, 'bottom')],
    #'SanityCheck': ['Put the santa in the box']
}


# consider recording rosbag, use ros ur_controllers
class TeleopNode:

    INIT_Q = [-0.461, -2.092, 1.844, -1.322, 4.718, -2.032]
    SCENE_BOUNDS = np.array([-0.70, -0.25, 0.03, -0.20, 0.25, 0.53])
    HOST = "192.168.1.179"
    CONTROL_FREQUENCY = 300.
    GRIPPER_VEL = 255
    GRIPPER_FORCE = 255
    FPS = 6.

    def __init__(self):
        self.observation_node = ROSObservationNode()
        flags = RTDEControl.FLAG_USE_EXT_UR_CAP | RTDEControl.FLAG_UPLOAD_SCRIPT
        self.rtde_c = RTDEControl(
            self.HOST,
            frequency=self.CONTROL_FREQUENCY,
            flags=flags
        )
        self.gripper = RobotiqCModelURCap(self.HOST)
        assert self.gripper.is_active()
        self.vr = vive.ViveController('/home/robot/leonid/teleop_dataset/vive_calibration.npz')
        # upd on episode init
        self.metatask = list(TASKS.keys())[0]
        self.task = TASKS[self.metatask][0]
        self._terminate = False
        self._prev_grip = 0
        self._displacement = np.zeros(3)

    def _measure_displacement(self):
        vr_state = self.vr.read_state()
        while vr_state.trigger < 1.:
            vr_state = self.vr.read_state()
        tcp_pose = self.get_obs()['tcp_pose']
        self._displacement = np.asarray(tcp_pose[:3]) - vr_state.position
        while vr_state.trigger > 0.:
            vr_state = self.vr.read_state()

    def initialize_episode(self):
        self.metatask = np.random.choice(list(TASKS.keys()))
        self.task = np.random.choice(TASKS[self.metatask])
        self._terminate = False
        self.rtde_c.moveJ(self.INIT_Q)
        self._prev_grip = 0
        self.gripper.move_and_wait_for_pos(self._prev_grip, 255, 255)
        print('Prepare the task: ', self.task)
        self._measure_displacement()

    def get_obs(self):
        obs = self.observation_node.get_observation()
        obs['description'] = self.task
        obs['is_terminal'] = self._terminate
        return obs

    def actuate(self, state: vive.ViveState):
        pos = state.position + self._displacement
        rot = state.rotation.as_rotvec()
        disc = 1  # 0.5
        grip = np.floor_divide(state.trigger, disc) * disc
        term = state.menu_button
        low, high = np.split(self.SCENE_BOUNDS, 2)
        pos = np.clip(pos, low, high)
        pose = np.concatenate([pos, rot])
        self._terminate = term > 0.5
        self.rtde_c.servoL(pose, 0., 0., 0.01, 0.1, 100)
        grip = int(255 * grip)
        if grip != self._prev_grip:
            self.gripper.move_and_wait_for_pos(grip, self.GRIPPER_VEL, self.GRIPPER_FORCE)
            self._prev_grip = grip

    def collect_demo(self):
        spf = 1. / self.FPS
        n_frames = 0
        no_state = 0
        obss = []
        cur_t = prev_t = time.time()
        self.initialize_episode()
        prev_state = self.vr.read_state()
        while not self._terminate:
            state = self.vr.read_state()
            if np.allclose(state.position, 0, atol=1e-8):
                self.rtde_c.servoStop()
                raise RuntimeError('Controller is lost.')
            if not np.allclose(state.position, prev_state.position, atol=0.07):
                no_state += 1
                if no_state > 10:
                    logging.info('Discontinuity encountered.')
                    break
                continue
            no_state = 0
            prev_state = state
            self.actuate(state)
            cur_t = time.time()
            if cur_t - prev_t > spf:
                prev_t = cur_t
                obs = self.get_obs()
                obss.append(obs)
                n_frames += 1
        self.rtde_c.servoStop()
        obs = self.get_obs()
        obss.append(obs)
        logging.info(f'Total frames: {n_frames}')
        return obss


def _read_trackpad(vr: vive.ViveController) -> int:
    val = 0
    while abs(val) < 0.75:
        val = vr.read_state().trackpad[0]
    val = 1 if val > 0 else 0
    print(val)
    return val


def main(dataset_path: str):
    teleop = TeleopNode()
    root = pathlib.Path(dataset_path).resolve()
    metatask = teleop.metatask
    task_dir = root / metatask
    task_dir.mkdir(exist_ok=True)
    while True:
        demo = teleop.collect_demo()
        assert teleop.metatask == metatask
        if any(map(lambda obs: obs['is_terminal'], demo[:-1])):
            print('Multiple termsigs')
            import pdb; pdb.set_trace()
        idx = len(list(task_dir.glob('*.pkl'))) + 1  # assumption on contiguous naming.
        print(f'Demo {idx} length: {len(demo)}. Is successful [0/1]?')
        if _read_trackpad(teleop.vr):
            with (task_dir / f'{idx:04d}.pkl').open(mode='wb') as f:
                pickle.dump(demo, f)


if __name__ == '__main__':
    rospy.init_node('teleop')
    logging.basicConfig(level=logging.INFO)
    main('/media/robot/Transcend/teleop_dataset')
