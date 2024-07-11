cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_utils)

add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

## Declare a C++ librar
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/pointcloud_utils.cpp
# )

add_executable(${PROJECT_NAME}_node src/surface_normals.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} pcl_visualization)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
