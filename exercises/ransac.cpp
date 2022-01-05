#include "ransac.hpp"
#include "matlab_like.hpp"
#include <algorithm>
#include <execution>

int count_inliers(
    Eigen::MatrixXd const &data,
    Eigen::MatrixXd const &poly,
    double max_noise)
{
    Eigen::MatrixXd error = (polyval(poly, data.topRows(1)).array() - data.bottomRows(1).array()).abs();
    return std::count_if(
        std::execution::par_unseq,
        error.data(),
        error.data() + error.size(),
        [max_noise](double err)
        { return err <= max_noise; });
}

void parabola_ransac(
    Eigen::MatrixXd const &data,
    double max_noise,
    Eigen::MatrixXd &best_guess_history,
    Eigen::MatrixXd &max_num_inliers_history)
{
    int num_iterations = 100;
    best_guess_history = Eigen::MatrixXd(3, num_iterations);
    max_num_inliers_history = Eigen::MatrixXd(1, num_iterations);

    int best_num_inliers = -1;
    Eigen::MatrixXd best_poly;

    for (int i = 0; i < num_iterations; ++i)
    {
        // chose 3 data points randomly from the data
        Eigen::MatrixXd sample = datasample(data, 3);

        // calculate a model guess from these data points
        Eigen::MatrixXd poly = polyfit(sample);

        // count inliers
        int num_inliers = count_inliers(data, poly, max_noise);

        // save the best guess
        if (num_inliers > best_num_inliers)
        {
            best_num_inliers = num_inliers;
            best_poly = poly;
        }
        max_num_inliers_history(0, i) = best_num_inliers;
        best_guess_history.col(i) = best_poly.transpose();
    }
}