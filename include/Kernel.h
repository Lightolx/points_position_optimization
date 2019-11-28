//
// Created by lightol on 2019/11/4.
//

#ifndef INC_3D_POINT_FILTERING_KERNEL_H
#define INC_3D_POINT_FILTERING_KERNEL_H


#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/opencv.hpp>

#include "common.h"

class Kernel
{
public:
    // Constructor
    Kernel() {}
    Kernel(const Eigen::Vector3d &point, double radius): mP(point) {}

    // Fill in the neighbor of the searching point
    void SetNeighborsCloud(const PointCloudT::Ptr &cloud);

    // Monotonically decreasing weighted function
    double kai(const Eigen::Vector3d &pt) const;

public:
    Eigen::Vector3d mP = Eigen::Vector3d::Zero();  // 核的球心
    double mRadius = 0.0;
    PointCloudT::Ptr mpNeighborsCloud = nullptr;  // spatial neighborhood of searching point

    Eigen::Vector3d mCentriod = Eigen::Vector3d::Zero();  // 核内所有点的重心
};

#endif //INC_3D_POINT_FILTERING_KERNEL_H
