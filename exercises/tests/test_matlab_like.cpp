#include "matlab_like.hpp"
#include <iostream>
#include <gtest/gtest.h>

TEST(MatlabLikeTest, polyval)
{
    Eigen::MatrixXd poly(1, 3);
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

TEST(MatlabLikeTest, datasample)
{
    for (int col = 3; col < 10; ++col)
    {
        Eigen::MatrixXd x = Eigen::MatrixXd::Random(3, col);
        Eigen::MatrixXd sampled_x = datasample(x, 3);
        std::cout << "\nx=\n" << x << std::endl;
        std::cout << "\nsampled_x=\n" << sampled_x << std::endl;
    }
}

TEST(MatlabLikeTest, random)
{
    Eigen::MatrixXd rand = random(3,5);
    EXPECT_EQ(rand.rows(), 3);
    EXPECT_EQ(rand.cols(), 5);
    for (auto x: rand.reshaped())
    {
        EXPECT_GT(x, 0.0);
        EXPECT_LT(x, 1.0);
    }
    std::cout << "\nrand=\n" << rand << std::endl;
}