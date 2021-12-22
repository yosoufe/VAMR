#pragma once

#include <Eigen/Core>

double polyval(
    Eigen::MatrixXd const &poly,
    double x);


Eigen::MatrixXd polyval(
    Eigen::MatrixXd const &poly,
    Eigen::MatrixXd const &x);