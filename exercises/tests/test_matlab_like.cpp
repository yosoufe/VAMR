#include "matlab_like.hpp"
#include <gtest/gtest.h>

TEST(MatlabLikeTest, polyval)
{
    Eigen::MatrixXd poly(3, 1);
    poly << 1, 2, 3;
    double res;
    res = polyval(poly, 1);
    EXPECT_EQ(res, 6);
    res = polyval(poly, 2);
    EXPECT_EQ(res, 11);
    Eigen::MatrixXd x{1, 2};
    x << 1, 2;
    auto res_matrix = polyval(poly, x);
    EXPECT_EQ(res_matrix(0, 0), 6);
    EXPECT_EQ(res_matrix(0, 1), 11);
    EXPECT_EQ(res_matrix.cols(), 2);
    EXPECT_EQ(res_matrix.rows(), 1);
}

TEST(MatlabLikeTest, polyfit)
{
    // create polynomial
    for (int n = 2; n < 10; ++n)
    {
        Eigen::MatrixXd poly_expected = Eigen::MatrixXd::Random(1, n);
        Eigen::MatrixXd x_s = Eigen::MatrixXd::Random(1, n);
        // sample data from the polynomial
        Eigen::MatrixXd y_s = polyval(poly_expected, x_s);
        Eigen::MatrixXd data(2, x_s.cols());

        data << x_s,
            y_s;

        // test the polyfit
        Eigen::MatrixXd poly_calculated = polyfit(data);

        EXPECT_EQ(poly_calculated.rows(), 1);
        EXPECT_EQ(poly_calculated.cols(), x_s.cols());
        EXPECT_TRUE(poly_calculated.isApprox(poly_expected, 1e-10));
    }
}