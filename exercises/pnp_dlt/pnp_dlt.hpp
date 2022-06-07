#pragma once

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

std::vector<Eigen::Matrix3Xd>
read_detected_corners(
    std::string file_path);

/**
 * @brief estimate the pose from the corners using DLT
 * 
 * @param detected_corners corners coordinates in pixels and in homogenous system,
 * Matrix is in shape of (3 X n), n is number of corners.
 * @param points_in_W points in world coordinate in shape of (3 x n).
 * @param K intrinsic camera matrix
 * @return Eigen::MatrixXd, the estimated pose of the camera in world coordinate. (4 x 4)
 */
Eigen::MatrixXd
estimate_pose_dlt(
    const Eigen::Matrix3Xd &detected_corners,
    const Eigen::MatrixXd &points_in_W,
    const Eigen::Matrix3d &K);

Eigen::MatrixXd
re_project_points(
    const Eigen::MatrixXd &points_in_W,
    const Eigen::MatrixXd &pose,
    const Eigen::Matrix3d &K);

void plot_trajectory_3d(
    const std::vector<Eigen::MatrixXd> &poses,
    const std::vector<cv::Mat> &images,
    const Eigen::MatrixXd &points_in_W,
    const std::string &output_file);