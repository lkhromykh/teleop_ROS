import random
import itertools
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from teleop.actuation import SocketActuation
from teleop.observables import ROSObservationNode


_BALLS = [f'{c} ball' for c in ('red', 'green', 'blue', 'yellow')]
_TOYS = ['red santa', 'grey seal', 'pink cat', 'white cat']
_CUBES = [f'{c} cube' for c in ('red', 'green', 'blue', 'yellow')]
TASKS = {
    'PickUp': [f'Pick up the {item}' for item in _TOYS],
    'PutInBox': [f'Put the {item} in the box' for item in _TOYS],
    'StackCubes': [f'Place the {c1} on top of the {c2}' for c1, c2 in itertools.combinations(_CUBES, 2)],
    'StackBlocks': [f'Stack {i} cubes' for i in range(2, 5)],
    'SlideCubes': [f'Slide the {c1} to the {c2}' for c1, c2 in itertools.combinations(_CUBES, 2)],
    'OpenDrawer': [f'Open the {shelf} drawer' for shelf in ('top', 'middle')],
}


class Scene:

    INIT_Q = [-0.461, -2.092, 1.844, -1.322, 4.718, -2.032]
    SCENE_BOUNDS = np.array([-0.70, -0.25, 0.03, -0.20, 0.25, 0.53])
    GRIPPER_VEL = 255
    GRIPPER_FORCE = 200

    def __init__(self,
                 tasks: Tuple[str] = tuple(TASKS.keys()),
                 real_time: bool = True,
                 ) -> None:
        assert len(tasks) and all(map(lambda t: t in TASKS.keys(), tasks))
        self.real_time = real_time
        self.tasks = tasks
        self._observation_node = ROSObservationNode()
        self._actuation_node = SocketActuation(host="192.168.1.179")
        # upd on episode init
        self.metatask = self.tasks[0]
        self.task = TASKS[self.metatask][0]
        self._termsig = False

    def initialize_episode(self) -> None:
        self.metatask = random.choice(self.tasks)
        self.task = random.choice(TASKS[self.metatask])
        self._termsig = False
        self._actuation_node.moveJ(self.INIT_Q)
        self._actuation_node.gripper_move_and_wait(0, self.GRIPPER_VEL, self.GRIPPER_FORCE)
        print('Prepare the task: ', self.task)
        if not self.real_time:
            input()

    def get_observation(self) -> Dict[str, np.ndarray]:
        obs = self._observation_node.get_observation()
        obs['description'] = np.asarray(self.task, dtype=np.dtype('U77'))
        obs['is_terminal'] = self.get_termination()
        for k, v in obs.items():
            obs[k] = np.asanyarry(v)
        return obs

    def get_termination(self) -> bool:
        return self._termsig

    def step(self, action: np.ndarray) -> None:
        # [xyz, yaw-pitch-roll, gripper, termsig]
        xyz, euler, grip, termsig = np.split(action, [3, 6, 7])
        rotvec = Rotation.from_euler(euler, 'ZYX').as_rotvec()
        low, high = np.split(self.SCENE_BOUNDS, 2)
        xyz = np.clip(xyz, low, high)
        tcp_pose = np.r_([xyz, rotvec])
        grip = int(255 * grip)
        self._termsig = termsig > 0.5
        if self.real_time:
            if self.get_termination():
                self._actuation_node.servoStop()
            else:
                self._actuation_node.servoL(tcp_pose, 0., 0., 0.01, 0.1, 100.)
        else:
            self._actuation_node.moveL(tcp_pose)
        self._actuation_node.gripper_move_and_wait(grip, self.GRIPPER_VEL, self.GRIPPER_FORCE)
