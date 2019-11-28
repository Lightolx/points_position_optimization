//
// Created by lightol on 2019/11/28.
//

#include <pcl/kdtree/kdtree_flann.h>
#include "utils.h"

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