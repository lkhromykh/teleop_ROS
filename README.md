# teleop_ROS
ROS1 workspace contents for UR5e & Robotiq 2F-85 teleoperation via HTC Vive controller. 

(TODO) As of now, portability is lacking. There is the one specific HOST that is able to use the project. Some relevant params are hardcoded.

## Requirements
* Linux setup with ROS.
* UR arm, Robotiq 2F-85 gripper, Azure Kinect, HTC Vive controller and base station.
* (TODO) Project partially relies on [ur_env](https://github.com/RQC-Robotics/ur5-env/tree/dev). Thus, it should be present in the PYTHONPATH.   


## Installation

1. Create a catkin workspace as usual (the following path serves as an example):
```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/teleop_ws/src
cd ~/teleop_ws/src
```

2. Populate the workspace: 
```bash
git clone https://github.com/lkhromykh/teleop_ROS.git
git submodule update --init --recursive
```
Consider reading the packages READMEs and install theirs requirements (should be already configured on the HOST).

3. Build the packages:
 ```bash
cd ~/teleop_ws
catkin_make
```

## Usage
1. Always make sure to use correct overlay 
```bash
source ~/teleop_ws/devel/setup.bash
```
2. Launch required nodes
```bash
roslaunch teleop teleop_setup.launch
```
3. Run a script
```bash
rosrun teleop vive_teleop.py
# OR
rosrun teleop env_server.py
```
