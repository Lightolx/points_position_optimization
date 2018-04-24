#include <iostream>
#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "kernel.h"

const double EPSILON = 0.000001;

using std::cout;
using std::endl;

struct SURFACE_FITTING_COST
{
    SURFACE_FITTING_COST(const Kernel &kernel): mKernel(kernel)
            {}

    template <typename T>
    bool operator()(const T* const position, T* residual) const
    {
        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);
        Eigen::Matrix<T, 3, 1> p0 = mKernel.mC.template cast<T>();
        Eigen::Matrix<T, 3, 1> l = mKernel.ml.template cast<T>();
        residual[0] = (p - p0).cross(l).norm();

        return true;
    }

    const Kernel mKernel;
};

int main()
{
    // Step0: Read in raw points
    PointCloudT::Ptr cloud0(new PointCloudT);
    pcl::io::loadPLYFile("/home/lightol/Lightol_prj/ClusterHDMap/result/ConditionClusteredLanes1_3.ply", *cloud0);

    // Step0.5: Radius outliers remove
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    // build the filter
    outrem.setInputCloud(cloud0);
    double radius = 1.0;
    outrem.setRadiusSearch(radius);
    outrem.setMinNeighborsInRadius(5);
    // apply filter
    PointCloudT::Ptr cloud(new PointCloudT);
    outrem.filter (*cloud);
    pcl::io::savePLYFile("outliers_removed.ply", *cloud);

    // Step1: For every point pi, Construct a kernel including it and its neighborhood
    // step1.1: Construct a pcl kd_tree for finding neighbors in sphere whose radius is 1.0m
    int numPts = cloud->points.size();
    std::vector<Eigen::Vector3d> vPoints;
    vPoints.resize(numPts);

    for (int i = 0; i < numPts; i++)
    {
        auto pt1 = cloud->points[i];
        vPoints[i] = Eigen::Vector3d(pt1.x, pt1.y, pt1.z);
    }

    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    std::vector<int> vIDs;
    std::vector<float> vSquaredDists;

    std::vector<Eigen::Vector3d> neighbors;
    std::vector<Kernel> kernels;
    kernels.resize(numPts);

    // step1.2: package pi with its neighbor as a kernel
    for (int i = 0; i < numPts; i++)
    {
        PointT sPoint = cloud->points[i];

        kdTree->radiusSearch(sPoint, radius, vIDs, vSquaredDists);
        if (vIDs.size() < 5)
        {
            continue;
        }
        // set center and size
        Kernel kernel(vPoints[i], radius);

        // set neighbors
        neighbors.clear();
        for (int id : vIDs)
        {
            neighbors.push_back(vPoints[id]);
        }
        kernel.setNeighbors(neighbors);
        kernel.lineFitting();

        // store this kernel
        kernels[i] = kernel;
    }

    // Step2: 对每一个3D点，搜索它附近的kernel，调整它的位置直到它到所有kernel的距离最小
    std::vector<Eigen::Vector3d> vNewPoints;
    vNewPoints.resize(numPts);
//    double distNeibor = 1.5;
    for (int i = 0; i < numPts; ++i) {
        Eigen::Vector3d pt = vPoints[i];
        double xyz[3] = {pt[0], pt[1], pt[2]};
        ceres::Problem problem;

        // add up all kernel's influence as residual blocks
        std::vector<double> residuals;
        residuals.reserve(kernels.size());
        std::vector<Eigen::Vector3d> cs;
        for (const Kernel &kernel : kernels)
        {
            Eigen::Vector3d p = kernel.mP;

            if ((pt - p).norm() > 1.5 * radius)
            {
                continue;
            }

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SURFACE_FITTING_COST, 1, 3>
                                             (new SURFACE_FITTING_COST(kernel)),
                                     nullptr, xyz);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 250;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // write back optimization results
        Eigen::Vector3d newPt = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
        vNewPoints[i] = newPt;
    }

    // Step5: Write back 3D points and export it
    for (int i = 0; i < cloud->size(); i++)
    {
        cloud->points[i].x = vNewPoints[i].x();
        cloud->points[i].y = vNewPoints[i].y();
        cloud->points[i].z = vNewPoints[i].z();
    }

    pcl::io::savePLYFile("opt.ply", *cloud);
//    std::ofstream fout("pt1.txt");
//    for (const Eigen::Vector3d &pt : vNewPoints)
//    {
//        fout << pt.x() << " " << pt.y() << " " << pt.z() << std::endl;
//    }
//    fout.close();
}