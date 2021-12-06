#include "two_view_geometry.hpp"
#include "counting_iterator.hpp"
#include <unsupported/Eigen/KroneckerProduct>

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
        std::execution::par, counting_iterator(0), p1.cols(),
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

double
dist_point_2_epipolar_line(
    Eigen::MatrixXd const &F,
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2)
{
    double cost = 0;
    int num_points = p1.cols();
    Eigen::MatrixXd homog_points(3, p1.cols() + p2.cols());
    homog_points << p1, p2;
    Eigen::MatrixXd epi_lines(3, p1.cols() + p2.cols());
    // epi_lines.block(0,0,3,p1.cols()) = F.transpose() * p2;
    // epi_lines.block(0,p1.cols(),3,epi_lines.cols()) = F * p1;
    epi_lines << F.transpose() * p2, F * p1;

    Eigen::MatrixXd denom = epi_lines.row(0).array().pow(2) + epi_lines.row(1).array().pow(2);
    assert(denom.rows() == 1);
    assert(denom.cols() == p1.cols() + p2.cols());

    cost = std::sqrt(
        (
            (epi_lines.array() * homog_points.array()).colwise().sum().pow(2) / denom.array())
            .sum() /
        num_points);
    return cost;
}

Eigen::MatrixXd
fundamental_eight_point(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2)
{
    Eigen::MatrixXd F(3, 3);
    Eigen::MatrixXd Q(p1.cols(), 9);
    std::for_each_n(
        std::execution::par, counting_iterator(0), p1.cols(),
        [&](int col)
        {
            int q_row = col;
            Eigen::MatrixXd row = Eigen::kroneckerProduct(p1.col(col), p2.col(col));
            Q.block(q_row, 0, 1, 9) = row.transpose();
        });
    // JacobiSVD vs BDCSVD
    Eigen::BDCSVD<Eigen::MatrixXd> svd_m(Q, Eigen::ComputeThinV);
    Eigen::MatrixXd svd_v = svd_m.matrixV();
    Eigen::MatrixXd temp_F = svd_v.col(svd_v.cols() - 1);
    F = temp_F.transpose().reshaped(3, 3).transpose();
    return F;
}