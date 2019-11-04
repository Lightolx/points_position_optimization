//
// Created by lightol on 2019/11/4.
//

#ifndef INC_3D_POINT_FILTERING_LINEKERNEL_H
#define INC_3D_POINT_FILTERING_LINEKERNEL_H

#include "Kernel.h"

class LineKernel: public Kernel {
public:
    LineKernel() {}
    LineKernel(const Eigen::Vector3d &point, double radius): Kernel(point, radius) {}

    // 把这一团点云进行直线拟合
    void lineFitting();

public:
    Eigen::Vector3d ml;         // 核内所有点拟合成的直线的方向向量
    Eigen::Vector3d mAnchor;    // 核内所有点拟合成的直线上一点
};


#endif //INC_3D_POINT_FILTERING_LINEKERNEL_H
