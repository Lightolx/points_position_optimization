//
// Created by lightol on 2019/11/28.
//

#ifndef INC_3D_POINT_FILTERING_UTILS_H
#define INC_3D_POINT_FILTERING_UTILS_H

#include "common.h"

// 模仿cloudcompare里面的subsample cloud by space
// cloud,       需要被降采样的原始点云
// resolution,  降采样的分辨率，也就是降采样之后任意两个点之间的距离都大于resolution
PointCloudT::Ptr SpaceSubsample(const PointCloudT::Ptr &cloud, double resolution);

// 对cloud_sampled里面的每个点粗略优化其位置，理论上以它为中心画一个球，球心会处在车道线点云的中心线上
void RoughOptimize(const PointCloudT::Ptr &cloud, PointCloudT::Ptr cloud_sampled, double radius);

#endif //INC_3D_POINT_FILTERING_UTILS_H

