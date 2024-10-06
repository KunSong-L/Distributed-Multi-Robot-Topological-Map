# To begin with
This is offical implementation of our work in RAL: [Multi-Robot Rendezvous in Unknown Environment With Limited Communication]([/guides/content/editing-an-existing-page](https://ieeexplore.ieee.org/document/10679913)).

This repo also contains some code for other work, and it is not formated in a clear way. In the following part, I will introduce how to use our code in a basic way.

## Overview
There are three different ROS packages in this repo.
- fht_map: our implementation of creating FHT-Map during exploration
- multi_fht_map: using FHT-Map to achieve rendezvous, which is our major work.
- turtle3sim: some necessary model/parameter/data of turtlebot for simulation

# Quick Start
## Platform
- Ubuntu 20.04
- ROS noetic

Using Ubuntu 20.04 is important, because we use python3.X to write our code.

## Denpendency
This part is the same with our previous work [FHT-Map](https://github.com/KunSong-L/FHT-Map).

### Python Packages
Some necessary packages are list below:
- Pytorch
- open3d for python
- opencv for pyhon

The detailed way for installing these packages will not be explained here.

### Cartographer
Cartographer is a 2D/3D map-building method. It provides the submaps' and the trajectories' information when building the map. We use the pose of robot and grid map constructed by Cartographer to build FHT-Map.

We suggest that you can refer to [Cartographer-for-SMMR](https://github.com/efc-robot/Cartographer-for-SMMR) to install the modified Cartographer to ```carto_catkin_ws```

and 

```
source /PATH/TO/CARTO_CATKIN_WS/devel_isolated/setup.bash
```

### Turtlebot3 Description
```
sudo apt install ros-noetic-turtlebot3*
sudo apt install ros-noetic-bfl
pip install future
sudo apt install ros-noetic-teb-local-planner
```

### Install Code for this work
```
mkdir ~/fht_map_ws/src && cd ~/fht_map_ws/src
git@github.com:KunSong-L/Distributed-Multi-Robot-Topological-Map.git
cd ..
catkin_make
source ./devel/setup.bash
```
We suggest that you can add *source ~/fht_map_ws/devel/setup.bash* to ~/.bashrc

## Start Simulation
Different simulation environments are provided in this repository.

To run our code, you need to open a simulation environment firstly.
```
roslaunch turtlebot3sim academy_env_three_robots.launch
```

Then, you need to start the 2-D SLAM and move-base module for turtlebot.
```
roslaunch turtlebot3sim three_robots_origin.launch
```

Finally, you can start the process of constructing FHT-Map.
```
roslaunch multi_fht_map multi_rendezvous_3robots.launch
```

### More Simulations
If you want to use other scenes, you can also lauch them easily.
```
roslaunch turtlebot3sim large_indoor_env_three_robots.launch
```

# Code Review
The most important file for this work is **multi_rendezvous.py** in *multi_fht_map/scripts/*.

# Citation
If you use this code for your research, please cite our papers. *https://arxiv.org/abs/2310.13899*

```
@article{song2024multi,
  title={Multi-Robot Rendezvous in Unknown Environment with Limited Communication},
  author={Song, Kun and Chen, Gaoming and Liu, Wenhang and Xiong, Zhenhua},
  journal={arXiv preprint arXiv:2405.08345},
  year={2024}
}
```