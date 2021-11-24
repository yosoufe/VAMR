#include "stereo_reconst.hpp"
#include "counting_iterator.hpp"
#include <limits>
#include <math.h>
#include <algorithm>
#include <execution>

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
                    // 3 least ssds are all smaller than
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
disparity_to_pointcloud(Eigen::MatrixXd const &disparity,
                        Eigen::MatrixXd const &K,
                        double baseline,
                        Eigen::MatrixXd const &left_img)
{
    auto num_valid_disp =
        disparity.size() -
        std::count(disparity.data(), disparity.data() + disparity.size(), 0.0);
    std::cout << "num_valid_disp " << num_valid_disp << std::endl;
    Eigen::MatrixXd point_cloud(3, num_valid_disp);

    Eigen::MatrixXd K_inv = K.inverse();
    std::cout << "K_inv: \n" << K_inv << std::endl;

    return point_cloud;
}
