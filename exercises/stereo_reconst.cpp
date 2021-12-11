#include "stereo_reconst.hpp"
#include "counting_iterator.hpp"
#include <limits>
#include <math.h>
#include <algorithm>
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

                    // cached now the SSDs for
                    // subpixel refinement laters
                    // if the disparity is not ignored
                    Eigen::MatrixXd Ys(3, 1);
                    Ys << SSDs[disparity - min_disp - 1],
                        SSDs[disparity - min_disp],
                        SSDs[disparity - min_disp + 1];

                    // reject outliers:
                    if (disparity == min_disp || disparity == max_disp)
                    {
                        // when best is at the limits,
                        // probably the best is out of the limit
                        // and we know the exact number
                        disparity = 0;
                    }
                    else
                    {
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
                    }

                    // refine in subpixel level
                    // with the values that are
                    // cached above
                    double refined_disp = static_cast<double>(disparity);
                    if (disparity != 0)
                    {
                        Eigen::MatrixXd A(3, 3);
                        A << 1, disparity - 1, (disparity - 1) * (disparity - 1),
                            1, disparity, disparity * disparity,
                            1, disparity + 1, (disparity + 1) * (disparity + 1);

                        Eigen::MatrixXd coeffs = A.inverse() * Ys;
                        refined_disp = (-coeffs(1, 0)) / (2.0 * coeffs(2, 0));
                    }

                    left_disp(row, col) = refined_disp;
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
    Eigen::MatrixXd p1_hat = K_inv * p1;
    A.block(0, 0, 3, 1) = p0_hat;
    A.block(0, 1, 3, 1) = -p1_hat;
    Eigen::MatrixXd A_t = A.transpose();
    Eigen::MatrixXd lambdas = (A_t * A).inverse() * A_t * b;
    double &lambda0 = lambdas(0, 0);
    res = lambda0 * p0_hat;
    return res;
}

size_t index_from_row_col(size_t row, size_t col, size_t n_cols)
{
    return row * n_cols + col;
}

Eigen::MatrixXd
disparity_to_pointcloud_serial_backup(
    Eigen::MatrixXd const &disparity,
    Eigen::MatrixXd const &K,
    double baseline,
    Eigen::MatrixXd const &left_img)
{
    auto num_valid_disp =
        disparity.size() -
        std::count(disparity.data(), disparity.data() + disparity.size(), 0.0);
    Eigen::MatrixXd point_cloud(4, num_valid_disp); // XYZ and intensity

    Eigen::MatrixXd K_inv = K.inverse();

    int point_idx = 0;

    for (size_t row = 0; row < disparity.rows(); ++row)
    {
        for (size_t col = 0; col < disparity.cols(); ++col)
        {
            if (disparity(row, col) == 0)
                continue;

            point_cloud.block(0, point_idx, 3, 1) =
                calculate_3d_from_disparity(K_inv,
                                            disparity,
                                            row,
                                            col,
                                            baseline);
            point_cloud(3, point_idx) = left_img(row, col);
            ++point_idx;
        }
    }
    assert(point_idx == num_valid_disp);
    return point_cloud;
}

Eigen::MatrixXd
disparity_to_pointcloud(
    Eigen::MatrixXd const &disparity,
    Eigen::MatrixXd const &K,
    double baseline,
    Eigen::MatrixXd const &left_img)
{
    Eigen::MatrixXd point_cloud(5, disparity.size()); // XYZ and intensity and valid
    Eigen::MatrixXd res;                              // XYZ and intensity

    Eigen::MatrixXd K_inv = K.inverse();

    // using for_each_n and execution::par speeds up the
    // complete program from taking 26.3 seconds
    // to 22.3 seconds (without viz)
    // although disparity_to_pointcloud_serial_backup is easier to write and read.

    std::for_each_n(
        std::execution::par, counting_iterator(0), disparity.rows(),
        [&K_inv, &disparity, &left_img, &point_cloud, baseline](size_t row)
        {
            std::for_each_n(
                std::execution::par, counting_iterator(0), disparity.cols(),
                [&K_inv, &disparity, &left_img, &point_cloud, row, baseline](size_t col)
                {
                    size_t idx = index_from_row_col(row, col, disparity.cols());
                    if (disparity(row, col) != 0)
                    {
                        point_cloud.block(0, idx, 3, 1) =
                            calculate_3d_from_disparity(K_inv,
                                                        disparity,
                                                        row,
                                                        col,
                                                        baseline);
                        point_cloud(3, idx) = left_img(row, col);
                        point_cloud(4, idx) = 1;
                    }
                    else
                    {
                        point_cloud(4, idx) = 0;
                    }
                });
        });

    std::vector<int> indices_to_keep;
    indices_to_keep.reserve(point_cloud.cols());
    for (int idx = 0; idx < point_cloud.cols(); ++idx)
    {
        if (point_cloud(4, idx) == 1)
            indices_to_keep.push_back(idx);
    }
    res = (point_cloud.block(0, 0, 4, point_cloud.cols()))(Eigen::all, indices_to_keep);
    return res;
}

void visualize_point_cloud(Eigen::MatrixXd const &point_cloud)
{
    using namespace cv;
    viz::Viz3d myWindow("Point Cloud");
    myWindow.setWindowSize(Size(1920, 1080));
    // location of the window,
    // Comment it out if you have only single monitor
    myWindow.setWindowPosition(Point(2560, 0));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(5));

    cv::Mat cv_cloud;
    Eigen::MatrixXd pc_transpose = point_cloud.transpose();
    cv::eigen2cv(pc_transpose, cv_cloud);
    cv_cloud = cv_cloud.reshape(4);

    cv::Mat cv_color;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigen_color =
        point_cloud.block(3, 0, 1, point_cloud.cols()).cast<unsigned char>().transpose();
    cv::eigen2cv(eigen_color, cv_color);

    size_t counter = 0;
    // std::cout << cv_cloud << std::endl;

    // https://answers.opencv.org/question/65569/cvviz-point-cloud/
    // std::cout << cv_color.channels() << std::endl;
    // std::cout << cv_color.type() << std::endl;
    // std::cout << cv_color.rows << std::endl;
    // std::cout << cv_color.cols << std::endl;
    // std::cout << cv_color.size() << std::endl;
    // std::cout << cv_cloud.size() << std::endl;
    cv::viz::WCloud cloud_widget{cv_cloud, cv_color};
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 4);

    std::stringstream str_;
    str_ << "point_cloud_" << ++counter;
    myWindow.showWidget(str_.str(), cloud_widget);

    myWindow.spin();
    myWindow.close();
}

void visualize_point_clouds(
    std::vector<Eigen::MatrixXd> const &point_clouds)
{
    using namespace cv;
    viz::Viz3d myWindow("Point Clouds");
    myWindow.setWindowSize(Size(1920, 1080));
    // location of the window,
    // Comment it out if you have only single monitor
    myWindow.setWindowPosition(Point(2560, 0));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(5));

    // myWindow.spinOnce(1, true);
    // myWindow.setViewerPose(
    //     Affine3d(
    //         Matx33d(
    //             0.9129581274202138, 0.006546646768235034, -0.4080007340599631,
    //             0.08046835856704371, 0.9773485475052571, 0.1957413087697381,
    //             0.4000403740210511, -0.2115347680771561, 0.8917515018477072),
    //         Vec3d(15.1419, -13.7171, -35.0615)));

    cv::viz::WCloudCollection cloud_collection_widget;
    cloud_collection_widget.setRenderingProperty(cv::viz::POINT_SIZE, 4);

    for (auto const &point_cloud : point_clouds)
    {
        cv::Mat cv_cloud;
        Eigen::MatrixXd pc_transpose = point_cloud.transpose();
        cv::eigen2cv(pc_transpose, cv_cloud);
        cv_cloud = cv_cloud.reshape(4);

        cv::Mat cv_color;
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigen_color =
            point_cloud.block(3, 0, 1, point_cloud.cols()).cast<unsigned char>().transpose();
        cv::eigen2cv(eigen_color, cv_color);

        cloud_collection_widget.addCloud(cv_cloud, cv_color);
    }

    myWindow.showWidget("point clouds", cloud_collection_widget);
    myWindow.spin();
    myWindow.close();
}

Eigen::MatrixXd
filter_point_cloud(
    std::vector<double> const &xlims,
    std::vector<double> const &ylims,
    std::vector<double> const &zlims,
    Eigen::MatrixXd const &lim_frame,
    Eigen::MatrixXd const &point_cloud)
{
    Eigen::MatrixXd res;
    std::vector<int> keep_indices;
    keep_indices.reserve(point_cloud.cols());

    Eigen::MatrixXd points_in_given_frame(point_cloud.rows(), point_cloud.cols());
    points_in_given_frame.block(0, 0, 3, points_in_given_frame.cols()) =
        lim_frame * point_cloud.block(0, 0, 3, point_cloud.cols());

    // visualize_point_cloud(points_in_given_frame);

    for (int idx = 0; idx < points_in_given_frame.cols(); ++idx)
    {
        auto const &x = points_in_given_frame(0, idx);
        auto const &y = points_in_given_frame(1, idx);
        auto const &z = points_in_given_frame(2, idx);

        if (x < xlims[0] || x > xlims[1] ||
            y < ylims[0] || y > ylims[1] ||
            z < zlims[0] || z > zlims[1])
            continue;

        keep_indices.push_back(idx);
    }
    res = point_cloud(Eigen::all, keep_indices);
    return res;
}