//
// Created by lightol on 3/26/18.
//

#ifndef INC_3D_POINT_FILTERING_KERNEL_H
#define INC_3D_POINT_FILTERING_KERNEL_H

#include <pcl/segmentation/sac_segmentation.h>
#include "Eigen/Eigen"
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class Kernel
{
public:
    // Constructor
    Kernel() {}
    Kernel(const Eigen::Vector3d &point, double radius): mP(point), mRadius(radius) {}

    // Fill in the neighbor of the searching point
    void setNeighbors(const std::vector<Eigen::Vector3d> &points);

    // Monotonically decreasing weighted function
    double kai(const Eigen::Vector3d &pt) const;

    // 把这一团点云进行直线拟合
    void lineFitting();

public:
    Eigen::Vector3d mP;  // searching point
    double mRadius;
    std::vector<Eigen::Vector3d> mvNeighbors;  // spatial neighborhood of searching point

    Eigen::Vector3d mC;  // 拟合成的内点的重心
    Eigen::Vector3d ml;  // 拟合成的直线的方向向量
};

void Kernel::setNeighbors(const std::vector<Eigen::Vector3d> &points)
{
    mvNeighbors.clear();
    mvNeighbors.assign(points.begin(), points.end());
}

double Kernel::kai(const Eigen::Vector3d &pt) const
{
    double dist = (pt - mP).norm();
    return 1 - 0.5 * dist / mRadius;
}

void Kernel::lineFitting()
{
    PointCloudT::Ptr cloud(new PointCloudT);
    int numPts = mvNeighbors.size();
    cloud->width = numPts;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->resize(cloud->width * cloud->height);
    for (int i = 0; i < cloud->size(); i++)
    {
        cloud->points[i].x = mvNeighbors[i].x();
        cloud->points[i].y = mvNeighbors[i].y();
        cloud->points[i].z = mvNeighbors[i].z();
    }

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    ml = Eigen::Vector3d(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
    ml.normalize();

    Eigen::Vector3d sumP = Eigen::Vector3d::Zero();
    pcl::PointXYZ pTmp;
    for (int id : inliers->indices)
    {
        pTmp = cloud->points[id];
        sumP += Eigen::Vector3d(pTmp.x, pTmp.y, pTmp.z);
    }

    mC = Eigen::Vector3d(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
}

#endif //INC_3D_POINT_FILTERING_KERNEL_H
