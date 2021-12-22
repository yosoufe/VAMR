#include "matlab_like.hpp"
#include "counting_iterator.hpp"

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