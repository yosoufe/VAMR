#include "stereo_reconst.hpp"
#include "counting_iterator.hpp"
#include <limits>
#include <math.h>
#include <algorithm>
#include <execution>
#include <cassert>
#include <opencv2/viz.hpp>

double squaredeuclidean(Eigen::MatrixXd const &img1,
                        Eigen::MatrixXd const &img2,
                        size_t col1, size_t row1,
                        size_t col2, size_t row2,
                        size_t patch_radius)
{
    double squaredeuclidean{0.0};
    for (int row_off_idx = -1 * int(patch_radius); row_off_idx <= int(patch_radius); ++row_off_idx)
    {
        for (int col_off_idx = -1 * int(patch_radius); col_off_idx <= int(patch_radius); ++col_off_idx)
        {
            squaredeuclidean += std::pow(
                img1(int(row1) + row_off_idx, int(col1) + col_off_idx) -
                    img2(int(row2) + row_off_idx, int(col2) + col_off_idx),
                2);
        }
    }
    return squaredeuclidean;
}

// get_disparity (0.131 s) is 6 time faster than get_disparity_backup (0.626s) on my machine
// This is the same as get_disparity_backup plus outlier rejection,
// and it uses `std::for_each_n` and
// `std::execution::par` instead of directly using loops.
Eigen::MatrixXd
get_disparity(Eigen::MatrixXd const &left_img,
              Eigen::MatrixXd const &right_img,
              size_t patch_radius,
              size_t min_disp,
              size_t max_disp)
{
    Eigen::MatrixXd left_disp =
        Eigen::MatrixXd::Zero(left_img.rows(),
                              left_img.cols());

    Index_t start_idx_row = patch_radius;
    Index_t num_iteration_row = (left_img.rows() - patch_radius) - start_idx_row;

    Index_t start_idx_col = patch_radius + max_disp;
    Index_t num_iteration_col = (left_img.cols() - patch_radius) - start_idx_col;

    std::for_each_n(
        std::execution::par, counting_iterator(start_idx_row), num_iteration_row,
        [=, &left_disp, &left_img, &right_img](size_t row)
        {
            std::for_each_n(
                std::execution::par, counting_iterator(start_idx_col), num_iteration_col,
                [=, &left_disp,
                 &left_img,
                 &right_img](size_t col)
                {
                    double min_ssd = std::numeric_limits<double>::max();
                    size_t min_idx;
                    std::vector<double> SSDs;
                    size_t disparity = 0;
                    for (size_t d = min_disp; d <= max_disp; ++d)
                    {
                        double ssd = squaredeuclidean(left_img, right_img,
                                                      col, row,
                                                      col - d, row,
                                                      patch_radius);
                        SSDs.push_back(ssd);
                        if (ssd < min_ssd)
                        {
                            min_ssd = ssd;
                            disparity = d;
                            min_idx = d - min_disp;
                        }
                    }

                    // reject outliers:
                    // reject the disparity if
                    // 3 least SSDs are all smaller than
                    // 1.5 * min_ssd.
                    // (of course min_ssd < 1.5 * min_ssd" )

                    // find the first 3 smallest items
                    // but not in order
                    std::nth_element(
                        std::execution::par,
                        SSDs.begin(),
                        SSDs.begin() + 3,
                        SSDs.end(),
                        std::less<double>());

                    size_t num_bad_samples = 0;
                    std::for_each(
                        SSDs.begin(), SSDs.begin() + 3,
                        [&num_bad_samples, min_ssd](double ssd)
                        {
                            if (ssd <= 1.5 * min_ssd)
                                ++num_bad_samples;
                        });
                    if (num_bad_samples == 3)
                        disparity = 0;

                    left_disp(row, col) = static_cast<double>(disparity);
                });
        });
    return left_disp;
}

Eigen::MatrixXd
get_disparity_backup(Eigen::MatrixXd const &left_img,
                     Eigen::MatrixXd const &right_img,
                     size_t patch_radius,
                     double min_disp,
                     double max_disp)
{
    Eigen::MatrixXd left_disp =
        Eigen::MatrixXd::Zero(left_img.rows(),
                              left_img.cols());

    for (size_t row = patch_radius;
         row < left_img.rows() - patch_radius;
         ++row)
    {
        for (size_t col = patch_radius + max_disp;
             col < left_img.cols() - patch_radius;
             ++col)
        {
            double min_dist = std::numeric_limits<double>::max();
            size_t disparity = 0;
            for (size_t d = min_disp; d <= max_disp; ++d)
            {
                double dist = squaredeuclidean(left_img, right_img,
                                               col, row,
                                               col - d, row,
                                               patch_radius);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    disparity = d;
                }
            }
            left_disp(row, col) = static_cast<double>(disparity);
        }
    }
    return left_disp;
}

Eigen::MatrixXd
calculate_3d_from_disparity(Eigen::MatrixXd const &K_inv,
                            Eigen::MatrixXd const &disparity,
                            int row,
                            int col,
                            double baseline)
{
    Eigen::MatrixXd res(3, 1);
    Eigen::MatrixXd p0(3, 1);
    Eigen::MatrixXd p1(3, 1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 2);
    p0 << col, row, 1.0;
    p1 << col - disparity(row, col), row, 1.0;
    Eigen::MatrixXd b(3, 1);
    b << baseline, 0, 0;
    Eigen::MatrixXd p0_hat = K_inv * p0;
    A.block(0, 0, 3, 1) = p0_hat;
    A.block(0, 1, 3, 1) = (-1.0) * (K_inv * p1);
    Eigen::MatrixXd lambdas = (A.transpose() * A).inverse() * A.transpose() * b;
    double &lambda0 = lambdas(0, 0);
    res = lambda0 * p0_hat;
    return res;
}

Eigen::MatrixXd
disparity_to_pointcloud(Eigen::MatrixXd const &disparity,
                        Eigen::MatrixXd const &K,
                        double baseline,
                        Eigen::MatrixXd const &left_img)
{
    auto num_valid_disp =
        disparity.size() -
        std::count(disparity.data(), disparity.data() + disparity.size(), 0.0);
    std::cout << "num_valid_disp " << num_valid_disp << std::endl;
    Eigen::MatrixXd point_cloud(4, num_valid_disp); // XYZ and intensity

    Eigen::MatrixXd K_inv = K.inverse();
    std::cout << "K_inv: \n"
              << K_inv << std::endl;

    int counter = 0;

    for (size_t row = 0; row < disparity.rows(); ++row)
    {
        for (size_t col = 0; col < disparity.cols(); ++col)
        {
            if (disparity(row, col) == 0)
                continue;

            point_cloud.block(0, counter, 3, 1) =
                calculate_3d_from_disparity(K_inv,
                                            disparity,
                                            row,
                                            col,
                                            baseline);
            point_cloud(3, counter) = left_img(row, col);
            ++counter;
        }
    }
    assert(counter == num_valid_disp);
    return point_cloud;
}

void visualize_point_cloud(Eigen::MatrixXd const &point_cloud)
{
    using namespace cv;
    viz::Viz3d myWindow("Point Cloud");
    myWindow.setWindowSize(Size(1920, 1080));
    // location of the window,
    // Comment it out if you have only single monitor
    myWindow.setWindowPosition(Point(2560, 0));

    cv::Mat cv_cloud;
    Eigen::MatrixXd pc_transpose = point_cloud.transpose();
    cv::eigen2cv(pc_transpose, cv_cloud);
    cv_cloud = cv_cloud.reshape(4);

    cv::Mat cv_color;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigen_color =
        point_cloud.block(3, 0, 1, point_cloud.cols()).cast<unsigned char>().transpose();
    cv::eigen2cv(eigen_color, cv_color);

    size_t counter = 0;

    // https://answers.opencv.org/question/65569/cvviz-point-cloud/
    // std::cout << cv_color.channels() << std::endl;
    // std::cout << cv_color.type() << std::endl;
    // std::cout << cv_color.rows << std::endl;
    // std::cout << cv_color.cols << std::endl;
    // std::cout << cv_color.size() << std::endl;
    // std::cout << cv_cloud.size() << std::endl;
    cv::viz::WCloud cloud_widget{cv_cloud, cv_color};
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 4 );

    std::stringstream str_;
    str_ << "point_cloud_" << ++counter;
    myWindow.showWidget(str_.str(), cloud_widget);

    myWindow.spin();
    myWindow.close();
}