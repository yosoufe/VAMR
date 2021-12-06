#include "two_view_geometry.hpp"
#include "counting_iterator.hpp"

Eigen::MatrixXd
cross_matrix(Eigen::MatrixXd const &vector3d)
{
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(3, 3);
    res(0, 1) = -vector3d(2, 0);
    res(0, 2) = vector3d(1, 0);
    res(1, 2) = -vector3d(0, 0);
    res(1, 0) = vector3d(2, 0);
    res(2, 0) = -vector3d(1, 0);
    res(2, 1) = vector3d(0, 0);
    return res;
}

Eigen::MatrixXd
linear_triangulation(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2,
    Eigen::MatrixXd const &M1,
    Eigen::MatrixXd const &M2)
{
    Eigen::MatrixXd P(4, p1.cols());

    std::for_each_n(
        std::execution::seq, counting_iterator(0), p1.cols(),
        [&](int col)
        {
            Eigen::MatrixXd A(6, 4);
            A.block(0, 0, 3, 4) = cross_matrix(p1.block(0, col, 3, 1)) * M1;
            A.block(3, 0, 3, 4) = cross_matrix(p2.block(0, col, 3, 1)) * M2;
            // JacobiSVD vs BDCSVD
            Eigen::BDCSVD<Eigen::MatrixXd> svd_m(A, Eigen::ComputeThinV);
            P.block(0, col, 4, 1) = svd_m.matrixV().col(3);
            P.block(0, col, 4, 1) = P.block(0, col, 4, 1).array() / P(3, col);
        });

    return P;
}