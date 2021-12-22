#pragma once

#include <Eigen/Core>

double polyval(
    Eigen::MatrixXd const &poly,
    double x);

Eigen::MatrixXd polyval(
    Eigen::MatrixXd const &poly,
    Eigen::MatrixXd const &x);

/**
 * @brief fit a polynomial with (n-1) order where n is the number of columns of x
 *
 * @param x (2, n) input data
 * @return Eigen::MatrixXd (1,n) coefficients of the fitted polynomial
 */
Eigen::MatrixXd polyfit(
    Eigen::MatrixXd const &x);