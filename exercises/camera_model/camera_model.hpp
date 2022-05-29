#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core.hpp>

Eigen::Matrix2Xd
distorted_pixel(const double k1,
                const double k2,
                const Eigen::Vector2d &principal_pt,
                const Eigen::Matrix2Xd &points);

Eigen::Matrix2Xd
project_2_camera_frame(const Eigen::Matrix3d &intrinsics,
                       const Eigen::Isometry3d &extrinsics,
                       const Eigen::Matrix3Xd &points);

cv::Mat
undistort_image(const cv::Mat &src_img,
                double d1,
                double d2,
                const Eigen::Vector2d &principal_pt);

namespace cuda
{
    cv::Mat
    undistort_image(const cv::Mat &src_img,
                    double d1,
                    double d2,
                    const Eigen::Vector2d &principal_pt);
}