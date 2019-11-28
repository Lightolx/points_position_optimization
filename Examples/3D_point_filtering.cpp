//
// Created by lightol on 2019/11/28.
//

#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "common.h"
#include "PointKernel.h"

using std::cout;
using std::endl;

int main() {
    // Step0: Read in raw points
    std::string filename = "../data/scissor.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(filename, *cloud);

    // Step1: For every point pi, Construct a kernel including it and its neighbors
    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    std::vector<int> vIDs;
    std::vector<float> vSquaredDists;
    double radius = 0.1;  // 这个值可以随机调节

    int numPts = cloud->points.size();
    cout << "numPts = " << numPts << endl;
    std::vector<PointKernel> vKernels;
    vKernels.resize(numPts);

    for (int i = 0; i < numPts; i++) {
        PointT sPoint = cloud->points[i];
        kdTree->radiusSearch(sPoint, radius, vIDs, vSquaredDists);
        // set center and size
        PointKernel kernel(Eigen::Vector3d(sPoint.x, sPoint.y, sPoint.z), radius);

        // set neighbors
        PointCloudT::Ptr cloud_neighbors(new PointCloudT);
        for (int id : vIDs) {
            cloud_neighbors->points.push_back(cloud->points[id]);
        }
        kernel.SetNeighborsCloud(cloud_neighbors);

        vKernels[i] = kernel;
    }

    // Step2: 对每一个采样3D点，搜索它附近的kernel，调整它的位置直到它到所有kernel的距离最小
    std::vector<Eigen::Vector3d> vNewPoints;
    vNewPoints.resize(numPts);
    for (int i = 0; i < numPts; ++i) {
        cout << "i = " << i << endl;
        PointT sPoint = cloud->points[i];
        Eigen::Vector3d p(sPoint.x, sPoint.y, sPoint.z);
        double xyz[3] = {p[0], p[1], p[2]};

        // add up all kernel's influence as residual blocks
        ceres::Problem problem;
        for (const PointKernel &kernel : vKernels) {
            if (!kernel.mpNeighborsCloud) {  // 这个点是个杂点，附近并没有足够的neighbor来形成一个kernel
                continue;
            }

            Eigen::Vector3d pt = kernel.mP;
            if ((pt - p).norm() > 1.5 * radius) {  // 由于距离太远，认为点p并不受由pt构成的kernel的影响
                continue;
            }

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Point_DIST_COST, 1, 3>
                                             (new Point_DIST_COST(kernel)), nullptr, xyz);
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

    pcl::io::savePLYFile("../result/scissor.ply", *cloud);
}
