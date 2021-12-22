#include "two_view_geometry.hpp"
#include <gtest/gtest.h>

/* to run this test in the build directory, run the following
 * command:
 *
 * ./tests/test_two_view_geometry --gtest_filter=Two_View_Geometry.linear_triangulation
 */
TEST(Two_View_Geometry, linear_triangulation)
{
    int N = 10;
    Eigen::MatrixXd P = Eigen::MatrixXd::Random(4, N);

    P.block(2, 0, 1, N) = (P.block(2, 0, 1, N) * 5.0).array() + 10.0;
    P.block(3, 0, 1, N) = Eigen::MatrixXd::Ones(1, N);

    Eigen::MatrixXd M1(3, 4);
    M1 << 500.0, 0.0, 320.0, 0.0,
        0.0, 500.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0;

    Eigen::MatrixXd M2(3, 4);
    M2 << 500.0, 0.0, 320.0, -100.0,
        0.0, 500.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0;

    Eigen::MatrixXd p1 = M1 * P;
    Eigen::MatrixXd p2 = M2 * P;

    Eigen::MatrixXd P_est = linear_triangulation(p1, p2, M1, M2);
    ASSERT_TRUE(P_est.cols() == P.cols()) << P_est.cols() << " vs " << P.cols();
    ASSERT_TRUE(P_est.rows() == P.rows()) << P_est.rows() << " vs " << P.rows();

    Eigen::MatrixXd diff = P_est - P;
    std::cout << "P_est - P = \n";
    std::cout << diff << std::endl;

    EXPECT_TRUE(diff.cwiseAbs().maxCoeff() < 1e-12);
}

/* to run this test in the build directory, run the following
 * command:
 *
 * ./tests/test_two_view_geometry --gtest_filter=Two_View_Geometry.eight_point
 */
TEST(Two_View_Geometry, eight_point)
{
    int N = 40;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(4, N);

    X.block(2, 0, 1, N) = (X.block(2, 0, 1, N) * 5.0).array() + 10.0;
    X.block(3, 0, 1, N) = Eigen::MatrixXd::Ones(1, N);

    Eigen::MatrixXd P1(3, 4);
    P1 << 500.0, 0.0, 320.0, 0.0,
        0.0, 500.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0;

    Eigen::MatrixXd P2(3, 4);
    P2 << 500.0, 0.0, 320.0, -100.0,
        0.0, 500.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0;

    Eigen::MatrixXd x1 = P1 * X;
    Eigen::MatrixXd x2 = P2 * X;

    double sigma = 1e-1;

    Eigen::MatrixXd noisy_x1 = x1 + sigma * Eigen::MatrixXd::Random(x1.rows(), x1.cols());
    Eigen::MatrixXd noisy_x2 = x2 + sigma * Eigen::MatrixXd::Random(x2.rows(), x2.cols());

    // Fundamental matrix estimation via the 8-point algorithm
    {
        // Estimate fundamental matrix
        // Call the 8-point algorithm on inputs x1,x2
        Eigen::MatrixXd F = fundamental_eight_point(x1, x2);

        // Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
        double cost_algebraic = algebraic_error(F, x1, x2);;
        double cost_dist_epi_line = dist_point_2_epipolar_line(F, x1, x2);

        std::cout << "Noise-free correspondences\n";
        std::cout << "Algebraic error: " << cost_algebraic << std::endl;
        std::cout << "Geometric error: " << cost_dist_epi_line << std::endl;

        EXPECT_TRUE(cost_algebraic < 1e-9);
        EXPECT_TRUE(cost_dist_epi_line < 1e-9);
    }

    {
        Eigen::MatrixXd F = fundamental_eight_point(noisy_x1, noisy_x2);
        double cost_algebraic = algebraic_error(F, noisy_x1, noisy_x2);
        double cost_dist_epi_line = dist_point_2_epipolar_line(F, noisy_x1, noisy_x2);

        std::cout << "Noisy correspondences, (sigma = " << sigma << "), with fundamental_eight_point\n";
        std::cout << "Algebraic error: " << cost_algebraic << std::endl;
        std::cout << "Geometric error: " << cost_dist_epi_line << std::endl;
    }

    {
        Eigen::MatrixXd Fn = fundamental_eight_point_normalized(noisy_x1, noisy_x2);
        double cost_algebraic = algebraic_error(Fn, noisy_x1, noisy_x2);
        double cost_dist_epi_line = dist_point_2_epipolar_line(Fn, noisy_x1, noisy_x2);

        std::cout << "Noisy correspondences, (sigma = " << sigma << "), with fundamental_eight_point_normalized\n";
        std::cout << "Algebraic error: " << cost_algebraic << std::endl;
        std::cout << "Geometric error: " << cost_dist_epi_line << std::endl;
    }
}