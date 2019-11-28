//
// Created by lightol on 2019/11/4.
//

#include <pcl/common/centroid.h>

#include "Kernel.h"

double Kernel::kai(const Eigen::Vector3d &pt) const
{
    double dist = (pt - mP).norm();
    return 1 - 0.5 * dist / mRadius;
}

void Kernel::SetNeighborsCloud(const PointCloudT::Ptr &cloud) {
    mpNeighborsCloud = cloud;

    Eigen::Vector4d centroid = Eigen::Vector4d::Zero();
    pcl::compute3DCentroid(*cloud, centroid);
    mCentriod = centroid.topRows(3);
}

