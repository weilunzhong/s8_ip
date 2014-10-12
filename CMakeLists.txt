cmake_minimum_required(VERSION 2.8.3)
project(s8_ip)
find_package(catkin REQUIRED COMPONENTS
  	sensor_msgs
	cv_bridge
	roscpp
	std_msgs
	image_transport
)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

add_executable(imageConverter src/image_converter.cpp)
target_link_libraries(imageConverter ${catkin_LIBRARIES})