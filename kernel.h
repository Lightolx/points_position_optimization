//
// Created by lightol on 3/26/18.
//

#ifndef INC_3D_POINT_FILTERING_KERNEL_H
#define INC_3D_POINT_FILTERING_KERNEL_H

#include "Eigen/Eigen"

class Kernel
{
public:
    Kernel() {}
    Kernel(const Eigen::Vector3d &point, const double &radius): sPoint(point), h(radius) {}
    void setNeighbors(const std::vector<Eigen::Vector3d> &points);
    void computeNormal();
    void computeC();
    void computeFai();

public:
    Eigen::Vector3d sPoint;  // searching point
    double h;                // kernel size
    std::vector<Eigen::Vector3d> neighbors;  // spatial neighborhood of searching point
    Eigen::Vector3d cPoint;  // weighted center point
    Eigen::Matrix3d C;       // weighted covariance matrix
    Eigen::Vector3d v1, v2;  // orthogonal vectors that consist of the fitted plane
    Eigen::Vector3d n;       // normal vector of the least_squared plane fitted to neighbors
    double sigma_X, sigma_Y; // function fai which is a 2D gaussian distribution
};

void Kernel::setNeighbors(const std::vector<Eigen::Vector3d> &points)
{
    neighbors.clear();
    neighbors.assign(points.begin(), points.end());
}

void Kernel::computeNormal()
{

}

void Kernel::computeC()
{

}

void Kernel::computeFai()
{

}

#endif //INC_3D_POINT_FILTERING_KERNEL_H
