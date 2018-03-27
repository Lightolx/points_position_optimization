cmake_minimum_required(VERSION 3.9)
project(3D_point_filtering)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(3D_point_filtering main.cpp kernel.h)

target_link_libraries(3D_point_filtering ${PCL_LIBRARIES})