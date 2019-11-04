cmake_minimum_required(VERSION 3.9)
project(3D_point_filtering)

set(CMAKE_CXX_STANDARD 11)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
add_definitions(${CERES_DEFINITIONS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(3D_point_filtering main.cpp Kernel.h Kernel.cpp PointKernel.h PointKernel.cpp LineKernel.h LineKernel.cpp)

target_link_libraries(3D_point_filtering ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})