//
// Created by lightol on 3/26/18.
//

#ifndef INC_3D_POINT_FILTERING_KERNEL_H
#define INC_3D_POINT_FILTERING_KERNEL_H

#include "Eigen/Eigen"
#include <opencv2/opencv.hpp>

class Kernel
{
public:
    // Constructor
    Kernel() {}
    Kernel(const Eigen::Vector3d &point, const double &radius): p_(point), h_(radius) {}

    // Fill in the neighbor of the searching point
    void setNeighbors(const std::vector<Eigen::Vector3d> &points);

    // Compute the covariance matrix
    void computeC();

    // Compute the normal of the fitted plane
    void computeNormal();

    // Compute the Gaussian distribution parameter sigma_X and sigma_Y
    void computeFai();

    // Compute the ellipsoid formed by this kernel, all paremeters included
    void computeEllip();

    // Monotonically decreasing weighted function
    double kai(const Eigen::Vector3d &pt) const;

public:
    Eigen::Vector3d p_;  // searching point
    double h_;                // kernel size
    std::vector<Eigen::Vector3d> neighbors_;  // spatial neighborhood of searching point
    Eigen::Vector3d c_;  // weighted center point
    Eigen::Vector3d v1_, v2_;  // orthogonal vectors that consist of the fitted plane
    Eigen::Vector3d n_;       // normal vector of the least_squared plane fitted to neighbors
    double sigma_X, sigma_Y; // function fai which is a 2D gaussian distribution
};

void Kernel::setNeighbors(const std::vector<Eigen::Vector3d> &points)
{
    neighbors_.clear();
    neighbors_.assign(points.begin(), points.end());
}

void Kernel::computeC()
{
//    C = Eigen::Matrix3d::Zero();
//
//    for (const Eigen::Vector3d &pt : neighbors)
//    {
//        C += kai(pt) * (pt - cPoint) * (pt - cPoint).inverse();
//    }
}

void Kernel::computeNormal()
{

}

void Kernel::computeFai()
{

}

double Kernel::kai(const Eigen::Vector3d &pt) const
{
    double dist = (pt - p_).norm();
    return 1 - 0.2 * dist / h_;
}

void Kernel::computeEllip()
{
    //step1: get the weighted center c_
    for (const Eigen::Vector3d &pt : neighbors_)
    {
        // todo:: there should be a weighted accumulate, not just a gravity center
//        c_ += kai(pt) * pt;
        c_ += pt;
    }
    c_ /= neighbors_.size();

    // step2: compute the covariance matrix C
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d &pt : neighbors_)
    {
        C += kai(pt) * (pt - c_) * (pt - c_).transpose();
    }

    // step3: eigenvalue decompose C to get the major and minor axis, the two vector form the
    //        x0y plane, cross them to get normal n as axis z
    Eigen::EigenSolver<Eigen::Matrix3d> es(C);
//    std::cout << C << std::endl;
    Eigen::Matrix3d V = es.pseudoEigenvectors();
//    std::cout << std::endl << V << std::endl;
    Eigen::Matrix3d D = es.pseudoEigenvalueMatrix();
//    std::cout << std::endl << D << std::endl;
    std::vector<double> eigVals(3);
    eigVals[0] = D(0, 0);
    eigVals[1] = D(1, 1);
    eigVals[2] = D(2, 2);

    // return value of pseudoEigenValue() is unordered, so manually find the max and the mid one
    std::vector<double>::iterator iter = std::max_element(eigVals.begin(), eigVals.end());
    int maxId = std::distance(eigVals.begin(), iter);
    iter = std::min_element(eigVals.begin(), eigVals.end());
    int minId = std::distance(eigVals.begin(), iter);
    int midId = 0;

    for (int i = 0; i < 3; ++i)
    {
        if (i != maxId && i != minId)
        {
            midId = i;
            break;
        }
    }

    v1_ = V.col(maxId);
    v2_ = V.col(midId);
    n_ = v1_.cross(v2_);

    // step4: project all neighbors to plane x0y, do a ellipse fitting to get the
    //        Gaussian distribution parameters sigma_X and sigma_Y
    std::vector<cv::Point2d> pts;
    pts.reserve(neighbors_.size());
    std::vector<double> Xs, Ys;
    Xs.reserve(neighbors_.size());
    Ys.reserve(neighbors_.size());

    for (const Eigen::Vector3d &pt : neighbors_)
    {
        pts.push_back(cv::Point2d((pt - c_).dot(v1_), (pt - c_).dot(v2_)));
        Xs.push_back(fabs((pt - c_).dot(v1_)));
        Ys.push_back(fabs((pt - c_).dot(v2_)));
    }

    // todo:: there should be a ellipse fitting, first extract the boundary, then do cv::fitEllipse
//    cv::RotatedRect rect = cv::fitEllipse(pts);
    iter = std::max_element(Xs.begin(), Xs.end());
    sigma_X = *iter / 10;
    iter = std::max_element(Ys.begin(), Ys.end());
    sigma_Y = *iter / 10;
}

#endif //INC_3D_POINT_FILTERING_KERNEL_H
