cmake_minimum_required(VERSION 3.9)
project(3D_point_filtering)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# PCL
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# Ceres
find_package(Ceres REQUIRED)
add_definitions(${CERES_DEFINITIONS})

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)  # 把src文件夹下的所有文件统一赋个名字　DIR_SRCS

set(3rd_LIBS
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${CERES_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Sophus_LIBRARIES}
    )

add_executable(FilterSolidLine Examples/FilterSolidLine.cpp ${DIR_SRCS})
target_link_libraries(FilterSolidLine ${3rd_LIBS})

add_executable(FilterDoubleYellw Examples/FilterDoubleYellow.cpp ${DIR_SRCS})
target_link_libraries(FilterDoubleYellw ${3rd_LIBS})

add_executable(3D_point_filtering Examples/3D_point_filtering.cpp ${DIR_SRCS})
target_link_libraries(3D_point_filtering ${3rd_LIBS})