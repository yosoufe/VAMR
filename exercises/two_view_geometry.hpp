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

/**
 * @brief The 8-point algorithm for the estimation of the fundamental matrix F
 *
 * The eight-point algorithm for the fundamental matrix with a posteriori
 * enforcement of the singularity constraint (det(F)=0).
 * Does not include data normalization.
 *
 * Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.
 *
 * @param p1 (3,N): homogeneous coordinates of 2-D points in image 1
 * @param p2 (3,N): homogeneous coordinates of 2-D points in image 2
 * @return Eigen::MatrixXd (3,3) : fundamental matrix
 */
Eigen::MatrixXd
fundamental_eight_point(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2);

/**
 * @brief Compute the point-to-epipolar-line distance
 *
 * @param F (3,3): Fundamental matrix
 * @param p1 (3,NumPoints): homogeneous coords of the observed points in image 1
 * @param p2 (3,NumPoints): homogeneous coords of the observed points in image 2
 * @return double sum of squared distance from points to epipolarlines normalized
 * by the number of point coordinates
 */
double
dist_point_2_epipolar_line(
    Eigen::MatrixXd const &F,
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2);