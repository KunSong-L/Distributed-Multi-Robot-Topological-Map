# MR-Topomap代码
## Dependency
install ceres python wrapper
https://github.com/Edwinem/ceres_python_bindings

## Run TopoExplore
```
{env_size}      = 'small' or 'large'
{number_robots} = 'single' or 'two' or 'three'
{method}        = 'rrt' or 'mmpf'
{suffix}        = 'robot' or 'robots' (be 'robot' when number_robots != 'single')
```

```
roslaunch turtlebot3sim {env_size}_env_{number_robots}_{suffix}.launch
roslaunch turtlebot3sim {number_robots}_{suffix}_origin.launch
roslaunch ros_topoexplore {number_robots}_{suffix}_topo.launch
```

for example 
### Single Robot in Small Env
```
roslaunch turtlebot3sim small_env_single_robot.launch
roslaunch turtlebot3sim single_robot_origin.launch
roslaunch ros_topoexplore single_robot_topo.launch
```

