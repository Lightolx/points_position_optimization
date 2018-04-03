#include <iostream>
#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "kernel.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

const double EPSILON = 0.000001;

using std::cout;
using std::endl;

struct SURFACE_FITTING_COST
{
    SURFACE_FITTING_COST(Eigen::Vector3d c, double h, Eigen::Vector3d v1, Eigen::Vector3d v2,
                         Eigen::Vector3d n, double sigmaX, double sigmaY): c_(c),
                         v1_(v1), v2_(v2), n_(n), h_(h), sigmaX_(sigmaX), sigmaY_(sigmaY)
            {}

    template <typename T>
    bool operator()(const T* const position, T* residual) const
    {
        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);
//        residual[0] = fai(p) * (pow(h_, 2) - pow((p - c_).dot(n_), 2));
//        residual[0] = fai(p) * pow((p - c_).dot(n_), 2);
//        T dist = (p - c_).norm();
//        residual[0] = dist * (p - c_).dot(n_);
        residual[0] = (p - c_).norm();
//        residual[1] = dist;

//        if (dist > 3*h_)
//        {
//            residual[1] = dist;
//        }
//        else
//        {
//            residual[1] = T(0);
//        }

//        residual[1] = T(1) / fai(p) * 0.1;

        return true;
    }

    // Compute the coefficient simulated by a gaussian distribution
    template <typename T>
    T fai(const  Eigen::Matrix<T, 3, 1> &p) const
    {
        T x = (p - c_).dot(v1_);
        T y = (p - c_).dot(v2_);
        T index = -0.5 * (pow(x, 2) / pow(sigmaX_, 2) + pow(y, 2) / pow(sigmaY_, 2));
        return 1 / (2 * M_PI * sigmaX_ * sigmaY_) * ceres::exp(index);
    }

    const Eigen::Vector3d c_;
    const Eigen::Vector3d v1_;
    const Eigen::Vector3d v2_;
    const Eigen::Vector3d n_;
    double h_;
    double sigmaX_;
    double sigmaY_;
};

double computeResidual(const Eigen::Vector3d &point, const Kernel &kernel)
{
    Eigen::Vector3d c = kernel.c_;
    Eigen::Vector3d v1 = kernel.v1_;
    Eigen::Vector3d v2 = kernel.v2_;
    Eigen::Vector3d n = kernel.n_;
    double sigma_X = kernel.sigma_X;
    double sigma_Y = kernel.sigma_Y;

    double len1 = v1.norm();
    double len2 = v2.norm();
    double len3 = n.norm();
    double x = (point - c).dot(v1);
    double y = (point - c).dot(v2);
    double index = -0.5 * (pow(x, 2) / pow(sigma_X, 2) + pow(y, 2) / pow(sigma_Y, 2));
    double fai = 1 / (2 * M_PI * sigma_X * sigma_Y) * ceres::exp(index);
    double dist = (point - c).dot(n);
//            double residual = fai * (pow(h, 2) - pow((point - c).dot(n), 2));
//    return pow((point - c).dot(n), 2);
//    return fai * (pow((point - c).dot(n), 2));
    return pow((point - c).norm(), 2);
}

int main()
{
    // Step0: Read in raw points

    std::ifstream fin("scissor.txt");
    std::string ptline;
    double x, y, z;
    std::vector<Eigen::Vector3d> points;

    while (getline(fin, ptline))
    {
        std::stringstream ss(ptline);
        ss >> x >> y >> z;
        points.push_back(Eigen::Vector3d(x,y,z));
    }

    /*
    PointCloudT::Ptr cloud(new PointCloudT);
//    if (pcl::io::loadPLYFile("dragon.ply", *cloud) < 0)
//    {
//        std::cerr << "no ply file found!" << std::endl;
//        abort();
//    }

    if (pcl::io::loadPCDFile("office.pcd", *cloud) < 0)
    {
        std::cerr << "no ply file found!" << std::endl;
        abort();
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(cloud->size());
    for (int i = 0; i < cloud->size(); i++)
    {
        points[i].x() = cloud->points[i].x;
        points[i].y() = cloud->points[i].y;
        points[i].z() = cloud->points[i].z;
    }*/


    // Step1: For every point pi, Construct a kernel including it and its neighborhood, version 1: fixed kernel size
    // step1.1: find pi's neighbor
    cout << points.size() << endl;
//    cout << points[554].x() << " " << points[554].y() << " " << points[554].z() << endl;
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->width = points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->resize(cloud->width * cloud->height);

    for (int i = 0; i < cloud->size(); i++)
    {
        cloud->points[i].x = points[i].x();
        cloud->points[i].y = points[i].y();
        cloud->points[i].z = points[i].z();
    }

    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>);
    kdTree->setInputCloud(cloud);
    float h = 0.04;                // kernel size
    std::vector<int> pointsID;
    std::vector<float> pointsSquaredDist;
    std::vector<Eigen::Vector3d> neighbors;
    std::vector<Kernel> kernels;
    kernels.reserve(cloud->size());

    // step1.2: packaging pi and its neighbor as a kernel
    int more = 0;
    int less = 0;
    for (int i = 0; i < cloud->size(); i++)
    {
        PointT sPoint = cloud->points[i];

        if (kdTree->radiusSearchT(sPoint, h, pointsID, pointsSquaredDist) > 5)
        {
            more++;
            // set center and size
            Kernel kernel(points[i], h);

            // set neighbors
            neighbors.clear();
            for (const int &id : pointsID)
            {
                neighbors.push_back(points[id]);
            }
            cout << "neighbor size is " << neighbors.size() << endl;
            kernel.setNeighbors(neighbors);

            // store this kernel
            kernels.push_back(kernel);
        }
        else
        {
            less++;
        }
    }
    cout << "more is " << more << ", less is " << less << endl;

    // Step2: Compute all parameter of the kernels, including v1_, v2_ and n_
    // Step2: Compute the normal of the least_squared plane fitted to neighbor pi
    // Step3: Compute the monotonically decreasing weighting function of every piece pi
    int i = 0;
    for (Kernel &kernel : kernels)
    {
        kernel.computeEllip();
//        // step3.1: compute the weighted covariance matrix C
//        kernel.computeC();
//
//        // step3.2: compute normal of the fitted plane
//        kernel.computeNormal();
//
//        // step3.3: compute the Gaussian function fai
//        kernel.computeFai();
//        cout << "neighbors " << i++ << endl;
    }

    // Step4: Accumulate all likelihood function to get the cost function, then optimize all
    //        points' position
    int it = 0;
//    cout << points.size();
    for (Eigen::Vector3d &point : points)
    {
        double xyz[3] = {point[0], point[1], point[2]};
        ceres::Problem problem;

        // add up all kernel's influence as residual blocks
//        cout << "kernels size is " << kernels.size() << endl;
        std::vector<double> residuals;
        residuals.reserve(kernels.size());
        std::vector<Eigen::Vector3d> cs;
        for (const Kernel &kernel : kernels)
        {
            Eigen::Vector3d c = kernel.c_;
            Eigen::Vector3d v1 = kernel.v1_;
            Eigen::Vector3d v2 = kernel.v2_;
            Eigen::Vector3d n = kernel.n_;
            double sigma_X = kernel.sigma_X;
            double sigma_Y = kernel.sigma_Y;

//            double X = (point - c).dot(v1);
//            double Y = (point - c).dot(v2);

//            if (fabs(X) > 3 * sigma_X || fabs(Y) > 3 * sigma_Y)
//            {
//                continue;
//            }de

            if ((point - c).norm() > 3 * h)
            {
                continue;
            }

            double residual = computeResidual(point, kernel);
            residuals.push_back(residual);

            // *******************************debug*********************************//
            /*double len1 = v1.norm();
            double len2 = v2.norm();
            double len3 = n.norm();
            double x = (point - c).dot(v1);
            double y = (point - c).dot(v2);
            double index = -0.5 * (pow(x, 2) / pow(sigma_X, 2) + pow(y, 2) / pow(sigma_Y, 2));
            double fai = 1 / (2 * M_PI * sigma_X * sigma_Y) * ceres::exp(index);
//            cout << "sigmaX is " << sigma_X << ", sigmaY is " << sigma_Y << endl;
            double dist = (point - c).dot(n);
//            double residual = fai * (pow(h, 2) - pow((point - c).dot(n), 2));
//            double residual = fai * (pow(h, 2) - pow((point - c).dot(n), 2));
            double residual = fai * (pow((point - c).dot(n), 2));
//            residual[0] = fai(p) * (pow(h_, 2) - pow((p - c_).dot(n_), 2));
//            double residual = fai * pow((point - c).dot(n), 2);
            if (residual < EPSILON)
            {
                residual = 0;
            }
//            cout << point << endl;
//            cout << c << endl;
//            cout << "dist is " << dist << endl;
            residuals.push_back(residual);*/
            /*if (j++ == 100 || j == 300 || j == 500 || j == 800)
            {
                std::cout << "fai is " << fai << ", residual is " << residual << std::endl;
//                int a = 1;
            }*/
            // *******************************debug*********************************//

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SURFACE_FITTING_COST, 1, 3>
                                    (new SURFACE_FITTING_COST(kernel.c_, h, v1, v2, n, sigma_X, sigma_Y)),
                                     nullptr, xyz);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//        options.minimizer_progress_to_stdout = true;
//        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "A point optimize done, i = " << i++ << std::endl;

        // write back optimization results
        Eigen::Vector3d opt = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
        if (i == 94)
        {
            cout << point << endl;
            cout << opt << endl;
        }
        cout << "diff is " << (opt - point).norm() << endl;

        /*
        std::sort(residuals.begin(), residuals.end() , std::greater<double>());
        for (auto residual : residuals)
        {
//                cout << residual << endl;
        }
        double error = 0.5 * std::accumulate(residuals.begin(), residuals.end(), 0.0);
//        cout << "total is: " << error << endl;
        Eigen::Vector3d Point(point);
         */

        point = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);

        /*
        residuals.clear();
        for (const Kernel &kernel : kernels)
        {
            Eigen::Vector3d c = kernel.c_;
            Eigen::Vector3d v1 = kernel.v1_;
            Eigen::Vector3d v2 = kernel.v2_;
            Eigen::Vector3d n = kernel.n_;
            double sigma_X = kernel.sigma_X;
            double sigma_Y = kernel.sigma_Y;

            double X = (Point - c).dot(v1);
            double Y = (Point - c).dot(v2);
            if (fabs(X) > 3 * sigma_X || fabs(Y) > 3 * sigma_Y)
            {
                continue;
            }
            double residual = computeResidual(point, kernel);
            residuals.push_back(residual);
        }
        std::sort(residuals.begin(), residuals.end() , std::greater<double>());
        for (auto residual : residuals)
        {
//                cout << "after optimizing, " << residual << endl;
        }
        double Error = 0.5 * std::accumulate(residuals.begin(), residuals.end(), 0.0);
        cout << "after, total is: " << Error << endl;
        cout << endl;

        if (it == 555)
        {
            cout << Point;
            cout << point;
        }*/

        cout << summary.BriefReport() << endl;

    }

    // Step5: Write back 3D points and export it
    /*for (int i = 0; i < cloud->size(); i++)
    {
        cloud->points[i].x = points[i].x();
        cloud->points[i].y = points[i].y();
        cloud->points[i].z = points[i].z();
    }

    pcl::io::savePLYFile("opt.ply", *cloud);*/
    std::ofstream fout("/media/psf/Home/Documents/MATLAB/pt1.txt");
    for (const Eigen::Vector3d &pt : points)
    {
        fout << pt.x() << " " << pt.y() << " " << pt.z() << std::endl;
    }
    fout.close();
}