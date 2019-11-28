//
// Created by lightol on 2019/11/4.
//

#ifndef INC_3D_POINT_FILTERING_POINTKERNEL_H
#define INC_3D_POINT_FILTERING_POINTKERNEL_H

#include "Kernel.h"

class PointKernel: public Kernel {
public:
    PointKernel() {}
    PointKernel(const Eigen::Vector3d &point, double radius): Kernel(point, radius) {}

private:
};

struct Point_DIST_COST {
    Point_DIST_COST(PointKernel kernel) : mKernel(std::move(kernel)) {}

    template<typename T>
    bool operator()(const T *const position, T *residual) const {
        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);
        Eigen::Matrix<T, 3, 1> p0 = mKernel.mCentriod.template cast<T>();
        // 3D点密度越高的核引力越大
        residual[0] = T(mKernel.mpNeighborsCloud->points.size()) * (p - p0).norm();

        return true;
    }

    const PointKernel mKernel;
};

#endif //INC_3D_POINT_FILTERING_POINTKERNEL_H
