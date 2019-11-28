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

#endif //INC_3D_POINT_FILTERING_LINEKERNEL_H
