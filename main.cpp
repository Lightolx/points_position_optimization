#include <iostream>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl-1.7/pcl/registration/icp.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "kernel.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct SURFACE_FITTING_COST
{
    SURFACE_FITTING_COST(Eigen::Vector3d p, Eigen::Vector3d c, double h,
                         Eigen::Vector3d v1, Eigen::Vector3d v2,
                         Eigen::Vector3d n, double sigmaX, double sigmaY): c_(c),
                         v1_(v1), v2_(v2), n_(n), h_(h), sigmaX_(sigmaX), sigmaY_(sigmaY)
            {}

    bool operator()(const double* const position, double* residual) const
    {
        Eigen::Vector3d p(position[0], position[1], position[2]);
//        residual[0] = fai(p) * (pow(h_, 2) - (p - c_).dot(n_));
        residual[0] = position[0] + position[1] + position[2];

        return true;
    }

    double fai(const Eigen::Vector3d &p) const
    {
        double x = (p - c_).dot(v1_);
        double y = (p - c_).dot(v2_);
        double index = -0.5 * (pow(x, 2) / pow(sigmaX_, 2) + pow(y, 2) / pow(sigmaY_, 2));
        return 1 / (2 * M_PI * sigmaX_ * sigmaY_) * ceres::exp(index);
    }

    const Eigen::Vector3d c_;
    const Eigen::Vector3d v1_;
    const Eigen::Vector3d v2_;
    const Eigen::Vector3d n_;
    double h_;
    double sigmaX_;
    double sigmaY_;
};

int main()
{
    // Step0: Read in raw points
    std::ifstream fin("pts.txt");
    std::string ptline;
    double x, y, z;
    std::vector<Eigen::Vector3d> points;

    while (getline(fin, ptline))
    {
        std::stringstream ss(ptline);
        ss >> x >> y >> z;
        points.push_back(Eigen::Vector3d(x,y,z));
    }

    // Step1: construct the neighborhood of every point pi, version 1: fixed kernel size
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->width = points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->resize(cloud->width * cloud->height);

    for (int i = 0; i < cloud->size(); i++)
    {
        cloud->points[i].x = points[i].x();
        cloud->points[i].y = points[i].y();
        cloud->points[i].z = points[i].z();
    }

    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    float h = 0.2;                // kernel size
    std::vector<int> pointsID;
    std::vector<float> pointsSquaredDist;
    std::vector<Eigen::Vector3d> neighbors;
    std::vector<Kernel> kernels;
    kernels.reserve(cloud->size());

    for (int i = 0; i < cloud->size(); i++)
    {
        PointT sPoint;

        if (kdTree->radiusSearchT(sPoint, h, pointsID, pointsSquaredDist) > 0)
        {
    // Step2: Compute the normal of the least_squared plane fitted to neighbor pi

            // step2.1: set center and size
            Kernel kernel(points[i], h);

            // step2.2: set neighbors
            neighbors.clear();
            for (const int &id : pointsID)
            {
                neighbors.push_back(points[id]);
            }
            kernel.setNeighbors(neighbors);

            // step2.3: store this kernel
            kernels.push_back(kernel);
        }
    }

    // Step3: Compute the monotonically decreasing weighting function of every piece pi
    for (Kernel &kernel : kernels)
    {
        // step3.1: compute normal of the fitted plane
        kernel.computeNormal();

        // step3.2: compute the weighted covariance matrix C
        kernel.computeC();

        // step3.3: compute the Gaussian function fai
        kernel.computeFai();
    }

    // Step4: Accumulate all likelihood function to get the cost function
    double xyz[3] = {points[0][0], points[0][1], points[0][2]};
    Eigen::Vector3d cPoint(points[0]);
    ceres::Problem problem;
    for (const Kernel &kernel : kernels)
    {
        Eigen::Vector3d p = kernel.sPoint;
        Eigen::Vector3d c = kernel.cPoint;
        Eigen::Vector3d v1 = kernel.v1;
        Eigen::Vector3d v2 = kernel.v2;
        Eigen::Vector3d n = kernel.n;
        double sigma_X = kernel.sigma_X;
        double sigma_Y = kernel.sigma_Y;
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SURFACE_FITTING_COST, 1, 3>
        (new SURFACE_FITTING_COST(p, c, h, v1, v2, n, sigma_X, sigma_Y)
        ), nullptr, xyz);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

}