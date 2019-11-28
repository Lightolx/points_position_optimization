#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/filters/random_sample.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <Eigen/Eigen>
#include <utility>
#include <ceres/ceres.h>

#include "common.h"
#include "utils.h"
#include "LineKernel.h"

using std::cout;
using std::endl;

int main() {
    std::string filepath = "/home/lightol/Desktop/line/*.ply";
    std::vector<cv::String> plyFilenames;
    cv::glob(filepath, plyFilenames);

    for (const auto &filename : plyFilenames) {
        std::string fileName = filename.substr(filename.find_last_of('/')+1);

        // Step0: Read in raw points
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

        // Step1: For every point pi, Construct a kernel including it and its neighbors
        pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
        kdTree->setInputCloud(cloud);
        std::vector<int> vIDs;
        std::vector<float> vSquaredDists;
        double radius = 1;  // 2m范围内车道线点全拿出来拟合直线

        int numPts_all = cloud->points.size();
        cout << "numPts_all = " << numPts_all << endl;
        std::vector<LineKernel> vKernels;
        vKernels.resize(numPts_all);

        for (int i = 0; i < numPts_all; i++) {
            PointT sPoint = cloud->points[i];
            kdTree->radiusSearch(sPoint, radius, vIDs, vSquaredDists);
            if (vIDs.size() < 600) {
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

        // Step2: 第二次降采样，免得优化采样点的位置时运算量过大
        PointCloudT::Ptr cloud_sampled = SpaceSubsample(cloud, 1);
//        pcl::io::savePLYFile("/home/lightol/Desktop/line/result/sampled_" + fileName, *cloud_sampled);

        // Step3: 对待优化的每一个3D点粗调其位置，降低等会精细调整的迭代次数
        RoughOptimize(cloud, cloud_sampled, 0.5);
        pcl::io::savePLYFile("/home/lightol/Desktop/line/result/optimized_" + fileName, *cloud_sampled);

        continue;
        // Step4: 对每一个采样3D点，搜索它附近的kernel，调整它的位置直到它到所有kernel的距离最小
        // 这种Line_Dist的效果反而不好
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
            cloud_sampled->points[i].x = vNewPoints[i].x();
            cloud_sampled->points[i].y = vNewPoints[i].y();
            cloud_sampled->points[i].z = vNewPoints[i].z();
        }

        pcl::io::savePLYFile("/home/lightol/Desktop/line/result/" + fileName, *cloud_sampled);
    }
}