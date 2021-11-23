#include "stereo_reconst.hpp"
#include <limits>
#include <math.h>

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

Eigen::MatrixXd
get_disparity(Eigen::MatrixXd const &left_img,
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
    // show(eigen_2_cv(left_disp * 255 / max_disp));
    return left_disp;
}