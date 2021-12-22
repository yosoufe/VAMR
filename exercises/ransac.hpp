#pragma once

#include <Eigen/Core>

/**
 * @brief parabola ransac
 *
 * @param data is 2xN with the data points given column-wise,
 * @param max_noise
 * @param best_guess_history (output) is 3xnum_iterations with the polynome coefficients
 * from polyfit of the BEST GUESS SO FAR at each iteration columnwise.
 * @param max_num_inliers_history (output) is 1xnum_iterations, with the inlier count of the
 * BEST GUESS SO FAR at each iteration.
 */
void parabola_ransac(
    Eigen::MatrixXd const &data,
    double max_noise,
    Eigen::MatrixXd &best_guess_history,
    Eigen::MatrixXd &max_num_inliers_history);