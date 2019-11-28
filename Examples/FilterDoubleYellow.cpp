//
// Created by lightol on 2019/11/5.
//

#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <Eigen/Eigen>
#include <utility>
#include <ceres/ceres.h>

#include "common.h"
#include "utils.h"
#include "PointKernel.h"

using std::cout;
using std::endl;

int main() {
    // Step0: Read in raw points
    std::string filename = "../data/doubleYellow.ply";
    std::string fileName = filename.substr(filename.find_last_of('/')+1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(filename, *cloud);


    // Step0.5: 第一次降采样，免得建立kernel集合时运算量过大
    {
        PointCloudT::Ptr cloud_temp(new PointCloudT);
        pcl::RandomSample<PointT> sor;      // 随机采样，降低点数的同时又能剔除一部分outlier
        sor.setInputCloud(cloud);
        sor.setSample(cloud->points.size() * 0.1);  // 只需要极少数的点，能拟合车道线就可以了
        sor.filter(*cloud_temp);

        *cloud = *cloud_temp;
    }

    // Step0.8: 统计滤波，滤狠一点，这样大多数outlier就被滤掉了
    {
        PointCloudT::Ptr cloud_temp(new PointCloudT);
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(10);
        sor.setStddevMulThresh(2.0);
        sor.filter(*cloud_temp);

        *cloud = *cloud_temp;
        pcl::io::savePLYFile("../data/result/filtered_" + fileName, *cloud);
    }


    // Step1: For every point pi, Construct a kernel including it and its neighbors
    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    std::vector<int> vIDs;
    std::vector<float> vSquaredDists;
    double radius = 0.1;  // 0.2m范围内的点认为还属于同一条车道线

    int numPts_all = cloud->points.size();
    cout << "numPts_all = " << numPts_all << endl;
    std::vector<PointKernel> vKernels;
    vKernels.resize(numPts_all);

    for (int i = 0; i < numPts_all; i++) {
        PointT sPoint = cloud->points[i];

        kdTree->radiusSearch(sPoint, radius, vIDs, vSquaredDists);
//            cout << "vIDs.size = " << vIDs.size() << endl;
//            if (vIDs.size() < 50) {
//                continue;
//            }
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

    // Step2: 第二次降采样，免得优化采样点的位置时运算量过大
    PointCloudT::Ptr cloud_sampled = SpaceSubsample(cloud, 0.4);
    pcl::io::savePLYFile("../data/result/sampled_" + fileName, *cloud_sampled);

    // Step3: 对待优化的每一个3D点粗调其位置，降低等会精细调整的迭代次数
    RoughOptimize(cloud, cloud_sampled, 0.2);
    pcl::io::savePLYFile("../data/result/optimized_" + fileName, *cloud_sampled);

    // Step4: 对每一个采样3D点，搜索它附近的kernel，调整它的位置直到它到所有kernel的距离最小
    int numPts = cloud_sampled->points.size();
    cout << "numPts = " << numPts << endl;
    std::vector<Eigen::Vector3d> vNewPoints;
    vNewPoints.resize(numPts);
    for (int i = 0; i < numPts; ++i) {
        cout << "i = " << i << endl;
        PointT sPoint = cloud_sampled->points[i];
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
        cloud_sampled->points[i].x = vNewPoints[i].x();
        cloud_sampled->points[i].y = vNewPoints[i].y();
        cloud_sampled->points[i].z = vNewPoints[i].z();
    }

    pcl::io::savePLYFile("../data/result/" + fileName, *cloud_sampled);
}
