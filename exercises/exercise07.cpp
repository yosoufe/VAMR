#include "utils.hpp"
#include "matlab_like.hpp"
#include "ransac.hpp"

void part_1()
{
    int num_inliers{20};
    int num_outliers{10};
    double noise_ratio{0.1};
    Eigen::MatrixXd poly = Eigen::MatrixXd::Random(3, 1); // random second order polynomial
    double extremum{-poly(1, 0) / (2 * poly(0, 0))};
    double xstart{extremum - 0.5};
    double lowest{polyval(poly, extremum)};
    double highest{polyval(poly, xstart)};
    double xspan{1.0};
    double yspan{highest - lowest};
    double max_noise{noise_ratio * yspan};

    Eigen::MatrixXd x_in = Eigen::MatrixXd::Random(1, num_inliers);
    x_in = x_in.array() + xstart;
    Eigen::MatrixXd y_in = polyval(poly, x_in);
    y_in = y_in + ((static_cast<Eigen::MatrixXd>(Eigen::MatrixXd::Random(y_in.rows(), y_in.cols())).array() - 0.5) * 2 * max_noise).matrix();

    Eigen::MatrixXd x_out = Eigen::MatrixXd::Random(1, num_outliers);
    x_out = x_out.array() + xstart;
    Eigen::MatrixXd y_out = Eigen::MatrixXd::Random(1, num_outliers);
    y_out = y_out.array() * yspan + lowest;

    Eigen::MatrixXd data{2, x_in.cols() + x_out.cols()};
    data << x_in, x_out,
        y_in, y_out;

    Eigen::MatrixXd best_guess_history, max_num_inliers_history;

    parabola_ransac(
        data,
        max_noise,
        best_guess_history,
        max_num_inliers_history);
}

int main()
{
    part_1();
    return 0;
}