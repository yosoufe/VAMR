#pragma once

#include "utils.hpp"

/**
 * @brief Linear Triangulation
 *
 * N is the number of points
 *
 * @param p1 (3,N): homogeneous coordinates of points in image 1
 * @param p2 (3,N): homogeneous coordinates of points in image 2
 * @param M1 (3,4): projection matrix corresponding to first image
 * @param M2 (3,4): projection matrix corresponding to second image
 * @return Eigen::MatrixXd (4,N): homogeneous coordinates of 3-D points
 */
Eigen::MatrixXd
linear_triangulation(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2,
    Eigen::MatrixXd const &M1,
    Eigen::MatrixXd const &M2);