#include "matlab_like.hpp"
#include <gtest/gtest.h>


TEST(RansacTest, polyval)
{
    Eigen::MatrixXd poly(3,1);
    poly << 1,2,3;
    double res;
    res = polyval(poly, 1);
    EXPECT_EQ(res, 6);
    res = polyval(poly , 2);
    EXPECT_EQ(res, 11);
    Eigen::MatrixXd x{1,2};
    x << 1,2;
    auto res_matrix = polyval(poly, x);
    EXPECT_EQ(res_matrix(0,0), 6);
    EXPECT_EQ(res_matrix(0,1), 11);
}