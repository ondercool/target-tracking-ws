# target-tracking-ws
The ROS2 repository developed for implementation of the proposed system in the article mentioned in the [target-tracking-paper](https://github.com/ondercool/target-tracking-paper.git) repository of mine.

# 2-DoF Vision-Based Target Tracking (ROS2 Workspace)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS 2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)

This repository contains the full ROS2 workspace and Gazebo simulation environment for the target tracking system described in the manuscript: *Physics-Informed Vision-Based Target Tracking: Dynamic ROI, Adaptive Noise Modeling, and Sensor-Only Evaluation*

## System Overview
This framework implements a resource-efficient classical vision stack (HSV/Hough) tightly coupled with a predicting Kalman filter, bypassing deep-learning latency for high-frequency closed-loop control.

## System Demonstration
*   **System Simulation:**
    * [Simulation Youtube Video Link](https://www.youtube.com/watch?v=0KLxBMH--0o)
*   **System Demo:**
    <video src="assets/system_demo_gif.mp4" autoplay loop muted playsinline width="100%"></video>
## Dependencies
*   Ubuntu 22.04
*   ROS2 Humble
*   Gazebo Ignition / classic
*   OpenCV

## Build & Execution

**1. Clone the repository and build the workspace:**
```bash
mkdir -p ~/tracking_ws/src
cd ~/tracking_ws/src
git clone https://github.com/ondercool/target-tracking-ws.git
cd ~/tracking_ws
colcon build
source install/setup.bash
```

**2. System Launch Instructions:**

***Note:*** Run all the listed bash commands in order on separate terminals, make sure you've built and sourced the workspace.
1. Launch the Simulation
```bash
ros2 launch vb_launch simulation.launch.py
```
2. Run the target tracker action server
```bash
ros2 run vb_actions lock_target_NCA_adaptive.py
```
3. Send a goal to the action server for initial target lock
```bash
ros2 action send_goal /vb_actions/lock_target_dynamic vb_actions/action/LockTargetDynamic "tracking_duration: 100.0"
```
4. Launch the test launch file to test system performance
- File arguments
    - **scenario**: specify a scenario number. use 1,2 or 4 
    - **model**: used for baseline vs. implementation comparison, use NCA_AUG for Adaptive Covariance Scaling
    
```bash
ros2 launch vb_controller test_target_lock_aug.launch.py scenario:=4 model:=NCA_AUG
```
5. (Optional) Run the key-vel controller to move the platform with keys
```bash
ros2 run vb_controller key_vel_controller.py
```
