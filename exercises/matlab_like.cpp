// #include <iostream>
#include "matlab_like.hpp"
#include "counting_iterator.hpp"
#include <Eigen/Dense>

double polyval(
    Eigen::MatrixXd const &poly,
    double x)
{
    int order = poly.size() - 1;
    double res = 0;
    int poly_idx = 0;
    while (order >= 0)
    {
        res += std::pow(x, order) * poly(poly_idx);
        order--;
        poly_idx++;
    }
    return res;
}

Eigen::MatrixXd polyval(
    Eigen::MatrixXd const &poly,
    Eigen::MatrixXd const &x)
{
    Eigen::MatrixXd res{1, x.cols()};
    std::for_each_n(
        std::execution::par_unseq, counting_iterator(0), x.cols(),
        [&res, &poly, &x](int col_idx)
        {
            res(0, col_idx) = polyval(poly, static_cast<double>(x(0, col_idx)));
        });
    return res;
}

Eigen::MatrixXd polyfit(
    Eigen::MatrixXd const &x)
{
    int order = x.cols()-1;
    Eigen::MatrixXd A(x.cols(), x.cols());
    for(int idx = 0; idx < x.cols(); ++idx)
    {
        int power = order - idx;
        A.block(0,idx,x.cols(), 1) = x.block(0,0, 1, x.cols()).transpose().array().pow(power);
    }

    Eigen::MatrixXd Y = x.block(1,0,1,x.cols()).transpose();
    Eigen::MatrixXd result = (A.inverse() * Y).transpose();

    return result;
}