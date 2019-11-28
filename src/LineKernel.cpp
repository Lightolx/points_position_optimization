//
// Created by lightol on 2019/11/4.
//

#include "LineKernel.h"

void LineKernel::lineFitting() {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);  // 假设车道线宽20cm
    seg.setInputCloud(mpNeighborsCloud);
    seg.segment(*inliers, *coefficients);

    ml = Eigen::Vector3d(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
    ml.normalize();
    mAnchor = Eigen::Vector3d(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
}