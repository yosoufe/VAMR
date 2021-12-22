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
        std::execution::par_unseq, counting_iterator(0), p1.cols(),
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
algebraic_error(
    Eigen::MatrixXd const &F,
    Eigen::MatrixXd const &x1,
    Eigen::MatrixXd const &x2)
{
    int N = x1.cols();
    return ((x2.array() * (F * x1).array()).matrix().colwise().sum()).norm() / std::sqrt(double(N));
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
        std::execution::par_unseq, counting_iterator(0), p1.cols(),
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
    F = temp_F.transpose().reshaped(3, 3);

    Eigen::BDCSVD<Eigen::MatrixXd> svd_F(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::MatrixXd sigmas = svd_F.singularValues().asDiagonal();
    sigmas(2, 2) = 0;
    F = svd_F.matrixU() * sigmas * svd_F.matrixV().transpose();
    return F;
}

Eigen::MatrixXd
matrix_std(Eigen::MatrixXd const &p)
{
    Eigen::MatrixXd res;
    res = ((p.array() - p.rowwise().mean().array()).square().rowwise().sum() / (p.cols())).sqrt();
    return res;
}

Eigen::MatrixXd
normalize_2d_pts(
    Eigen::MatrixXd const &p,
    Eigen::MatrixXd &T)
{
    int N = p.cols();
    Eigen::MatrixXd res;
    Eigen::MatrixXd eucl_ps = Eigen::MatrixXd::Ones(3, N);
    eucl_ps.block(0, 0, 2, N) = p.block(0, 0, 2, N).array().rowwise() / p.row(2).array();

    Eigen::MatrixXd eucl_pts_2 = eucl_ps.block(0, 0, 2, N);

    Eigen::Vector2d mean = eucl_pts_2.rowwise().mean();

    Eigen::MatrixXd pts_centered = eucl_pts_2.colwise() - mean;
    double sigma = std::sqrt(
        pts_centered.array().square().colwise().sum().mean());

    double s = sqrt(2) / sigma;

    T.resize(3, 3);

    T << s, 0, -s * mean(0),
        0, s, -s * mean(1),
        0, 0, 1;

    res = T * eucl_ps;
    return res;
}

Eigen::MatrixXd
fundamental_eight_point_normalized(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2)
{
    Eigen::MatrixXd T1(3, 3);
    Eigen::MatrixXd normalized_p1 = normalize_2d_pts(p1, T1);
    Eigen::MatrixXd T2(3, 3);
    Eigen::MatrixXd normalized_p2 = normalize_2d_pts(p2, T2);

    Eigen::MatrixXd F_normalized = fundamental_eight_point(normalized_p1, normalized_p2);
    Eigen::MatrixXd F = T2.transpose() * F_normalized * T1;
    return F;
}

Eigen::MatrixXd
estimate_essential_matrix(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2,
    Eigen::MatrixXd const &K1,
    Eigen::MatrixXd const &K2)
{
    Eigen::MatrixXd F = fundamental_eight_point_normalized(p1, p2);
    Eigen::MatrixXd E = K2.transpose() * F * K1;
    return E;
}

void decompose_essential_matrix(
    Eigen::MatrixXd const &E,
    std::vector<Eigen::MatrixXd> &R,
    Eigen::MatrixXd &u3)
{
    Eigen::MatrixXd W(3, 3);
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    // JacobiSVD vs BDCSVD
    Eigen::BDCSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinV | Eigen::ComputeThinU);
    u3 = svd.matrixU().col(2);
    float u3_norm = u3.norm();
    if (u3_norm != 0) u3 /= u3_norm;
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd U = svd.matrixU();

    Eigen::MatrixXd R_;
    R_ = U * W * V.transpose();
    if (R_.determinant() < 0) R_ *= -1;
    R.push_back(R_);

    R_ = U * W.transpose() * V.transpose();
    if (R_.determinant() < 0) R_ *= -1;
    R.push_back(R_);
    return;
}

void disambiguate_relative_pose(
    std::vector<Eigen::MatrixXd> const &Rots,
    Eigen::MatrixXd const &u3,
    Eigen::MatrixXd const &points0_h,
    Eigen::MatrixXd const &points1_h,
    Eigen::MatrixXd const &K1,
    Eigen::MatrixXd const &K2,
    Eigen::MatrixXd &R,
    Eigen::MatrixXd &T)
{
    std::vector<double> u3_factors = {1.0, -1.0};
    double correct_u3_factor = 0;
    int correct_rot_idx = -1;

    Eigen::MatrixXd M1 = K1 * Eigen::MatrixXd::Identity(3,4);
    Eigen::MatrixXd M2_(3,4);
    int max_num_points_in_front_of_cameras = 0;
    for (auto const & u3_factor : u3_factors)
    {
        for (int idx = 0; idx < Rots.size() ; ++ idx)
        {
            M2_ << Rots[idx], u3_factor * u3 ;
            Eigen::MatrixXd M2 = K1 * M2_;
            Eigen::MatrixXd points_3d_1 = linear_triangulation(points0_h, points1_h, M1, M2);
            // project 3d points in both cameras
            Eigen::MatrixXd points_3d_2 = M2_ * points_3d_1;
            Eigen::MatrixXd Zs_1 = points_3d_1.row(2);
            Eigen::MatrixXd Zs_2 = points_3d_2.row(2);
            int num_points_in_front_of_cameras = ( Zs_1.array()  > 0).count() + ( Zs_2.array()  > 0).count();
            if (num_points_in_front_of_cameras > max_num_points_in_front_of_cameras)
            {
                max_num_points_in_front_of_cameras = num_points_in_front_of_cameras;
                correct_rot_idx = idx;
                correct_u3_factor = u3_factor;
            }
        }
    }
    R = Rots[correct_rot_idx];
    T = correct_u3_factor * u3;
    return;
}
