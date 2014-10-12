cmake_minimum_required(VERSION 2.8.3)
project(s8_IP)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  ras_arduino_msgs
)
catkin_package(
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)