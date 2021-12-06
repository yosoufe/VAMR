#include "two_view_geometry.hpp"
#include <gtest/gtest.h>

/* to run this test in the build directory, run the following
 * command:
 *
 * ./tests/two_view_geometry_tests --gtest_filter=Two_View_Geometry.linear_triangulation
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