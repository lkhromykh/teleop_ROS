#!/usr/bin/env python
import rospy
import tree
import dm_env
import numpy as np

from ur_env.remote import RemoteEnvServer

from teleop.scene import Scene


class Environment(dm_env.Environment):
    """dm_env.Environment wrapper for Scene."""

    def __init__(self, scene: Scene):
        self.scene = scene

    def reset(self) -> dm_env.TimeStep:
        self.scene.initialize_episode()
        obs = self.scene.get_observation()
        return dm_env.restart(obs)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self.scene.actuate(action)
        obs = self.scene.get_observation()
        if obs["is_terminal"]:
            reward = self.get_reward()
            return dm_env.termination(reward, obs)
        return dm_env.transition(0., obs)

    def get_reward(self) -> float:
        while True:
            try:
                print("The task is done? 0/1")
                rew = float(input())
                assert rew in (0., 1.)
                return rew
            except (ValueError, AssertionError) as exc:
                rospy.info("Wrong input: %s" % exc)
                continue

    def observation_spec(self):
        def np_to_spec(x): return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(np_to_spec, self.scene.get_observation())

    def action_spec(self):
        f32 = np.float32
        rot_lim = np.full((3,), np.pi)
        xyz_min, xyz_max = np.split(self.scene.BOUNDS, 2)
        low = np.concatenate([xyz_min, -rot_lim, [0., 0.]], dtype=f32)
        high = np.concatenate([xyz_max, rot_lim, [1., 1.]], dtype=f32)
        description = "[x, y, z, rx, ry, rz, grip, term]"
        return dm_env.specs.BoundedArray((8,), f32, low, high, name=description)


def main(tasks: str, port: int = 5555):
    scene = Scene(tasks=tasks, real_time=False)
    env = Environment(scene)
    env = RemoteEnvServer(env, ("", port))
    env.run()


if __name__ == "__main__":
    rospy.init_node("teleop")
    tasks_ = ("PutInBox",)
    try:
        main(tasks_)
    except rospy.ROSInterruptException:
        pass

