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


#endif //INC_3D_POINT_FILTERING_POINTKERNEL_H
