#include "utils.hpp"
#include "matlab_like.hpp"
#include "ransac.hpp"
#include <vector>

#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>

std::vector<double>
eigen_2_vector(Eigen::MatrixXd const &eigen)
{
    return std::vector<double>(eigen.data(), eigen.data() + eigen.size());
}

void part_1()
{
    int num_inliers{20};
    int num_outliers{10};
    double noise_ratio{0.1};
    Eigen::MatrixXd poly = random(1, 3); // random second order polynomial
    double extremum{-poly(0, 1) / (2 * poly(0, 0))};
    double xstart{extremum - 0.5};
    double lowest{polyval(poly, extremum)};
    double highest{polyval(poly, xstart)};
    double xspan{1.0};
    double yspan{highest - lowest};
    double max_noise{noise_ratio * yspan};

    Eigen::MatrixXd x_in = random(1, num_inliers);
    x_in = x_in.array() + xstart;
    Eigen::MatrixXd y_in = polyval(poly, x_in);
    y_in = y_in + (static_cast<Eigen::MatrixXd>(random(y_in.rows(), y_in.cols())).array() - 0.5).matrix() * 2 * max_noise;

    Eigen::MatrixXd x_out = random(1, num_outliers);
    x_out = x_out.array() + xstart;
    Eigen::MatrixXd y_out = random(1, num_outliers);
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

    std::cout << "\n max_noise = " << max_noise << std::endl;

    std::cout << "\n best guess\n"
              << best_guess_history.rightCols(1).transpose() << std::endl;
    std::cout << "\n reference poly\n"
              << poly << std::endl;

    auto axes_iter = CvPlot::plot(
        std::vector<double>(
            max_num_inliers_history.data(),
            max_num_inliers_history.data() +
                max_num_inliers_history.size()));
    cv::imshow("iterations", axes_iter.render(480, 640));

    Eigen::MatrixXd x_eigen;
    {
        Eigen::VectorXd x;
        x.setLinSpaced(100, xstart, xstart + 1);
        x_eigen = x;
        x_eigen = x_eigen.transpose();
    }

    std::vector<double> x1;
    x1 = eigen_2_vector(x_eigen);

    CvPlot::Axes axes = CvPlot::makePlotAxes();
    // history "blue"
    for (int idx = 0; idx < best_guess_history.cols(); ++idx)
    {
        axes.create<CvPlot::Series>(x1, eigen_2_vector(polyval(best_guess_history.col(idx).transpose(), x_eigen)), "-g");
    }

    // original "green"
    axes.create<CvPlot::Series>(x1, eigen_2_vector(polyval(poly, x_eigen)), "-b");

    // data points
    axes.create<CvPlot::Series>(eigen_2_vector(data.row(0)), eigen_2_vector(data.row(1)), "ob");

    // best estimate, red
    axes.create<CvPlot::Series>(x1, eigen_2_vector(polyval(best_guess_history.rightCols(1).transpose(), x_eigen)), "-r");
    axes.setXLim({xstart, xstart + 1});
    axes.setYLim({lowest - max_noise, highest + max_noise});
    CvPlot::show("ploynomials", axes, 480, 640);
}

int main()
{
    part_1();
    return 0;
}