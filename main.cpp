#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <Eigen/Eigen>
#include <utility>
#include <ceres/ceres.h>

#include "LineKernel.h"

const double EPSILON = 0.000001;

using std::cout;
using std::endl;

PointCloudT::Ptr SpaceSubsample(const PointCloudT::Ptr &cloud, double resolution);

void RoughOptimize(const PointCloudT::Ptr &cloud, PointCloudT::Ptr cloud_sampled, double radius);

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
    std::string filepath = "/home/lightol/Desktop/result/Line/SolidLine/plys/*.ply";
    std::vector<cv::String> plyFilenames;
    cv::glob(filepath, plyFilenames);

    for (const auto &filename : plyFilenames) {
        std::string fileName = filename.substr(filename.find_last_of('/')+1);
        cout << "filename is " << fileName << endl;
        if (fileName == "all.ply") {
            continue;
        }
        // Step0: Read in raw points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPLYFile(filename, *cloud);
//        pcl::io::loadPLYFile("/home/lightol/Desktop/result/Line/SolidLine/solid5.ply", *cloud);

        // Step0.5: 第一次降采样，免得建立kernel集合时运算量过大
        {
            PointCloudT::Ptr cloud_temp(new PointCloudT);
//            pcl::VoxelGrid<PointT> sor;  // 注意用PointT而不是PointCloudT
//            sor.setInputCloud(cloud);
//            sor.setLeafSize(0.2, 0.2, 0.05);  // 车道线的采样密度暂定为每隔0.2m取一个点

            pcl::RandomSample<PointT> sor;      // 随机采样，降低点数的同时又能剔除一部分outlier
            sor.setInputCloud(cloud);
            sor.setSample(cloud->points.size() * 0.1);  // 只需要极少数的点，能拟合车道线就可以了
            sor.filter(*cloud_temp);

//            pcl::UniformSampling<PointT> sor;
//            sor.setInputCloud(cloud);
//            sor.setRadiusSearch(0.1);
//            pcl::PointCloud<int> ptIdxs;
//            sor.compute(ptIdxs);
//            pcl::copyPointCloud(*cloud, ptIdxs.points, *cloud_temp);

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
//            cout << "vIDs.size = " << vIDs.size() << endl;
//            if (vIDs.size() < 50) {
//                continue;
//            }
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
        /*
        PointCloudT::Ptr cloud_sampled(new PointCloudT);
        pcl::VoxelGrid<PointT> sor;  // 注意用PointT而不是PointCloudT
        sor.setInputCloud(cloud);
        cout << "minum pts in voxel = " << sor.getMinimumPointsNumberPerVoxel() << endl;
        cout << "negative is " << sor.getFilterLimitsNegative() << endl;
        sor.setFilterLimitsNegative(false);
        sor.setLeafSize(0.8, 0.8, 0.1);  // 车道线的采样密度暂定为每隔0.2m取一个点

//        pcl::RandomSample<PointT> sor;      // 随机采样，降低点数的同时又能剔除一部分outlier
//        sor.setInputCloud(cloud);
//        sor.setSample(cloud->points.size() * 0.01);  // 只需要极少数的点，能拟合车道线就可以了
        sor.filter(*cloud_sampled);

//        pcl::UniformSampling<PointT> sor;
//        sor.setInputCloud(cloud);
//        sor.setRadiusSearch(0.1);
//        pcl::PointCloud<int> ptIdxs;
//        sor.compute(ptIdxs);
//        pcl::copyPointCloud(*cloud, ptIdxs.points, *cloud_temp);

//        *cloud = *cloud_temp;
         */

        PointCloudT::Ptr cloud_sampled = SpaceSubsample(cloud, 0.5);
//        pcl::io::savePLYFile("/home/lightol/Desktop/result/Line/SolidLine/ThinLines/sampled_" + fileName, *cloud_sampled);

        // Step3: 对待优化的每一个3D点粗调其位置，降低等会精细调整的迭代次数
        RoughOptimize(cloud, cloud_sampled, 0.5);
        pcl::io::savePLYFile("/home/lightol/Desktop/result/Line/SolidLine/ThinLines/optimized_" + fileName, *cloud_sampled);

        continue;
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

        pcl::io::savePLYFile("/home/lightol/Desktop/result/Line/SolidLine/ThinLines/" + fileName, *cloud_sampled);
    }
}

// 模仿cloudcompare里面的subsample cloud by space
// cloud,       需要被降采样的原始点云
// resolution,  降采样的分辨率，也就是降采样之后任意两个点之间的距离都大于resolution
PointCloudT::Ptr SpaceSubsample(const PointCloudT::Ptr &cloud, double resolution) {
    int numPts = cloud->points.size();
    std::vector<Eigen::Vector3d> vPts(numPts);
    for (int i = 0; i < numPts; ++i) {
        PointT pt = cloud->points[i];
        vPts[i] = Eigen::Vector3d(pt.x, pt.y, pt.z);
    }

    std::vector<Eigen::Vector3d> vNewPts;
    for (const auto &p : vPts) {
        bool bSelected = true;
        for (const auto &q : vNewPts) {
            if ((q - p).norm() < resolution) {
                bSelected = false;
                break;
            }
        }

        if (bSelected) {
            vNewPts.push_back(p);
        }
    }

    PointCloudT::Ptr cloud_sampled(new PointCloudT);
    PointT pt_temp;
    for (const auto &pt : vNewPts) {
        pt_temp.x = pt.x();
        pt_temp.y = pt.y();
        pt_temp.z = pt.z();
        cloud_sampled->points.push_back(pt_temp);
    }

    cloud_sampled->height = 1;
    cloud_sampled->width = vNewPts.size();

    return cloud_sampled;
}


// 对cloud_sampled里面的每个点粗略优化其位置，理论上以它为中心画一个球，球心会处在车道线点云的中心线上
void RoughOptimize(const PointCloudT::Ptr &cloud, PointCloudT::Ptr cloud_sampled, double radius) {
    pcl::KdTreeFLANN<PointT> kdTree;
    kdTree.setInputCloud(cloud);

    std::vector<int> ptIdxs;
    std::vector<float> ptSquareDists;
    for (PointT &pt: cloud_sampled->points) {
        kdTree.radiusSearch(pt, radius, ptIdxs, ptSquareDists);

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int id : ptIdxs) {
            PointT temp_pt = cloud->points[id];
            centroid += Eigen::Vector3d(temp_pt.x, temp_pt.y, temp_pt.z);
        }
        centroid /= ptIdxs.size();

        pt.x = centroid.x();
        pt.y = centroid.y();
        pt.z = centroid.z();
    }
}