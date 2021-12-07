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
fundamental_eight_point_normalized(
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

/**
 * @brief normalizes 2D homogeneous points
 *
 * Function translates and normalizes a set of 2D homogeneous points
 * so that their centroid is at the origin and their mean distance from
 * the origin is sqrt(2).
 *
 * @param p 3xN array of 2D homogeneous coordinates
 * @param T output, The 3x3 transformation matrix, pts_tilda = T*pts
 * @return Eigen::MatrixXd 3xN array of transformed 2D homogeneous coordinates.
 */
Eigen::MatrixXd
normalize_2d_pts(
    Eigen::MatrixXd const &p,
    Eigen::MatrixXd &T);

double
algebraic_error(
    Eigen::MatrixXd const &F,
    Eigen::MatrixXd const &x1,
    Eigen::MatrixXd const &x2);

/**
 * @brief estimates the essential matrix
 *
 * @param p1 (3,N): homogeneous coordinates of 2-D points in image 1
 * @param p2 (3,N): homogeneous coordinates of 2-D points in image 2
 * @param K1 (3,3): calibration matrix of camera 1
 * @param K2 (3,3): calibration matrix of camera 2
 * @return Eigen::MatrixXd (3,3) : fundamental matrix
 */
Eigen::MatrixXd
estimate_essential_matrix(
    Eigen::MatrixXd const &p1,
    Eigen::MatrixXd const &p2,
    Eigen::MatrixXd const &K1,
    Eigen::MatrixXd const &K2);

/**
 * @brief Given an essential matrix, compute the camera motion, i.e.,  R and T such
 * that E ~ T_x R
 *
 * @param E (3,3) : Essential matrix
 * @param R vector of (3,3) matrices : output, the two possible rotations
 * @param u3 (3,1) : output, a vector with the translation information
 */
void decompose_essential_matrix(
    Eigen::MatrixXd const &E,
    std::vector<Eigen::MatrixXd> &R,
    Eigen::MatrixXd &u3);

/**
 * @brief finds the correct relative camera pose (among
 * four possible configurations) by returning the one that yields points
 * lying in front of the image plane (with positive depth).
 *
 * R and T is calculated where [R|t] = T_C1_C0 = T_C1_W is a transformation that maps points
 * from the world coordinate system (identical to the coordinate system of camera 0)
 * to camera 1.
 *
 * @param Rots vector of (3,3) matrices : the two possible rotations calculated by decompose_essential_matrix
 * @param u3 a 3x1 vector with the translation information calculated by decompose_essential_matrix
 * @param points0_h 3xN homogeneous coordinates of point correspondences in image 1
 * @param points1_h 3xN homogeneous coordinates of point correspondences in image 2
 * @param K0 3x3 calibration matrix for camera 1
 * @param K1 3x3 calibration matrix for camera 2
 * @param R output: 3x3 the correct rotation matrix
 * @param T output: 3x1 the correct translation vector
 */
void disambiguate_relative_pose(
    std::vector<Eigen::MatrixXd> const &Rots,
    Eigen::MatrixXd const &u3,
    Eigen::MatrixXd const &points0_h,
    Eigen::MatrixXd const &points1_h,
    Eigen::MatrixXd const &K0,
    Eigen::MatrixXd const &K1,
    Eigen::MatrixXd &R,
    Eigen::MatrixXd &T);