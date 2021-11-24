#pragma once

#include <array>
#include "utils.hpp"

/**
 * @brief left_img and right_img are both H x W and this should return a H x W
 * matrix containing the disparity d for each pixel of left_img. Set
 * disp_img to 0 for pixels where the SSD and/or d is not defined, and for d
 * estimates rejected in Part 2. patch_radius specifies the SSD patch and
 * each valid d should satisfy min_disp <= d <= max_disp.
 *
 * @param left_img
 * @param right_img
 * @param patch_radius
 * @param min_disp
 * @param max_disp
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd
get_disparity(Eigen::MatrixXd const &left_img,
              Eigen::MatrixXd const &right_img,
              size_t patch_radius,
              size_t min_disp,
              size_t max_disp);

Eigen::MatrixXd
get_disparity_backup(Eigen::MatrixXd const &left_img,
                     Eigen::MatrixXd const &right_img,
                     size_t patch_radius,
                     double min_disp,
                     double max_disp);

Eigen::MatrixXd
disparity_to_pointcloud(Eigen::MatrixXd const &disparity,
                        Eigen::MatrixXd const &K,
                        double baseline,
                        Eigen::MatrixXd const &left_img);