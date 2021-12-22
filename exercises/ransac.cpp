#include "ransac.hpp"

void parabola_ransac(
    Eigen::MatrixXd const &data,
    double max_noise,
    Eigen::MatrixXd &best_guess_history,
    Eigen::MatrixXd &max_num_inliers_history)
{
    int num_iterations = 100;
    best_guess_history = Eigen::MatrixXd(3, num_iterations);
    max_num_inliers_history = Eigen::MatrixXd(1, num_iterations);

    // chose 3 data points randomly from the data

    // calculate a model guess from these data points

    // count inliers

    // save the best guess
}