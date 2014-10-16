cmake_minimum_required(VERSION 2.8.3)
project(s8_ip)
find_package(catkin REQUIRED COMPONENTS
  	sensor_msgs
	cv_bridge
	roscpp
	std_msgs
	image_transport
	message_generation
)

add_message_files(FILES
	distPose.msg
)

generate_messages(DEPENDENCIES
	std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime
)
include_directories(${catkin_INCLUDE_DIRS})


add_executable(imageConverter src/image_converter.cpp)
target_link_libraries(imageConverter ${catkin_LIBRARIES})



add_executable(objectTracker src/object_tracker.cpp)
target_link_libraries(objectTracker ${catkin_LIBRARIES})

