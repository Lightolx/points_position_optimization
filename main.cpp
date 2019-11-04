#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Eigen>
#include <utility>
#include <ceres/ceres.h>

#include "LineKernel.h"

const double EPSILON = 0.000001;

using std::cout;
using std::endl;

struct LINE_DIST_COST {
    LINE_DIST_COST(LineKernel kernel): mKernel(std::move(kernel)) {}

    template <typename T>
    bool operator()(const T* const position, T* residual) const {
        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);
        Eigen::Matrix<T, 3, 1> p0 = mKernel.mAnchor.template cast<T>();
        Eigen::Matrix<T, 3, 1> l = mKernel.ml.template cast<T>();
        residual[0] = (p - p0).cross(l).norm();

        return true;
    }

    const LineKernel mKernel;
};

int main() {
    // Step0: Read in raw points
    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::io::loadPLYFile("/home/lightol/Desktop/result/Line/SolidLine/plys/solid37.ply", *cloud);
//    pcl::io::loadPLYFile("/home/lightol/Desktop/solid33.ply", *cloud);

    // Step0.5: 降采样，免得运算量过大
    pcl::VoxelGrid<PointT> sor;  // 注意用PointT而不是PointCloudT
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.03, 0.03, 0.03);
    PointCloudT::Ptr cloud_temp(new PointCloudT);
    sor.filter(*cloud_temp);
    *cloud = *cloud_temp;

    // Step1: For every point pi, Construct a kernel including it and its neighbors
    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    std::vector<int> vIDs;
    std::vector<float> vSquaredDists;
    double radius = 0.5;  // 1m范围内车道线点全拿出来拟合直线

    int numPts = cloud->points.size();
    std::vector<LineKernel> vKernels;
    vKernels.resize(numPts);

    for (int i = 0; i < numPts; i++) {
        PointT sPoint = cloud->points[i];

        kdTree->radiusSearch(sPoint, radius, vIDs, vSquaredDists);
//        cout << "vIDs.size = " << vIDs.size() << endl;
        if (vIDs.size() < 100) {
            continue;
        }
        // set center and size
        LineKernel kernel(Eigen::Vector3d(sPoint.x, sPoint.y, sPoint.z), radius);

        // set neighbors
        PointCloudT::Ptr cloud_neighbors(new PointCloudT);
        for (int id : vIDs) {
            cloud_neighbors->points.push_back(cloud->points[id]);
        }
        kernel.SetNeighborsCloud(cloud_neighbors);
        kernel.lineFitting();

        vKernels[i] = kernel;
    }

    // Step2: 对每一个3D点，搜索它附近的kernel，调整它的位置直到它到所有kernel的距离最小
    std::vector<Eigen::Vector3d> vNewPoints;
    vNewPoints.resize(numPts);
    cout << "numPts = " << numPts << endl;
    for (int i = 0; i < numPts; ++i) {
        cout << "i = " << i << endl;
        PointT sPoint = cloud->points[i];
        Eigen::Vector3d p(sPoint.x, sPoint.y, sPoint.z);
        double xyz[3] = {p[0], p[1], p[2]};

        // add up all kernel's influence as residual blocks
        ceres::Problem problem;
        for (const LineKernel &kernel : vKernels) {
            if (!kernel.mpNeighborsCloud) {  // 这个点是个杂点，附近并没有足够的neighbor来形成一个kernel
                continue;
            }

            Eigen::Vector3d pt = kernel.mP;
            if ((pt - p).norm() > 1.5 * radius) {  // 由于距离太远，认为点p并不受由pt构成的kernel的影响
                continue;
            }

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LINE_DIST_COST, 1, 3>
                                             (new LINE_DIST_COST(kernel)), nullptr, xyz);
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
    for (int i = 0; i < numPts; i++) {
        cloud->points[i].x = vNewPoints[i].x();
        cloud->points[i].y = vNewPoints[i].y();
        cloud->points[i].z = vNewPoints[i].z();
    }

    pcl::io::savePLYFile("/home/lightol/Desktop/result/Line/SolidLine/ThinLines/opt.ply", *cloud);
}