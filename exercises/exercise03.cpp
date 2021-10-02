#include "utils.hpp"
#include "folder_manager.hpp"
#include <cassert>

Eigen::MatrixXd sobel_x_kernel()
{
    return Eigen::Matrix3d(
        {{-1.0, 0.0, 1.0},
         {-2.0, 0.0, 2.0},
         {-1.0, 0.0, 1.0}});
}

Eigen::MatrixXd sobel_y_kernel()
{
    return Eigen::Matrix3d(
        {{-1.0, -2.0, -1.0},
         {0.0, 0.0, 0.0},
         {1.0, 2.0, 1.0}});
}

Eigen::MatrixXd correlation(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel)
{
    assert(kernel.cols() == kernel.rows());
    assert(kernel.cols() % 2 == 1);

    size_t kernel_r = kernel.cols() / 2;
    size_t kernel_s = kernel.cols();

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    for (size_t v = kernel_r; v < input.rows() - kernel_r; v++)
    {
        for (size_t u = kernel_r; u < input.cols() - kernel_r; u++)
        {
            auto element_wise_prod = input.block(v - kernel_r, u - kernel_r, kernel_s, kernel_s).array() * kernel.array();
            res(v, u) = element_wise_prod.sum();
        }
    }
    return res;
}

auto shi_tomasi(const cv::Mat &img, size_t patch_size)
{
    cv::Mat img_double;
    Eigen::MatrixXd eigen_img;
    cv::cv2eigen(img, eigen_img);
}

auto harris(cv::Mat &img, size_t patch_size, double kappa)
{
    cv::Mat img_double;
    Eigen::MatrixXd eigen_img;
    cv::cv2eigen(img, eigen_img);

    auto I_x = correlation(eigen_img, sobel_x_kernel());
    auto I_y = correlation(eigen_img, sobel_y_kernel());
    // visualize_matrix_as_image(eigen_img);
    // visualize_matrix_as_image(I_x);
    // visualize_matrix_as_image(I_y);
    auto I_xx = I_x.array().square().matrix();
    auto I_yy = I_y.array().square().matrix();
    auto I_xy = (I_x.array() * I_y.array()).matrix();
    auto sI_xx = correlation(I_xx, Eigen::MatrixXd::Ones(patch_size, patch_size));
    auto sI_yy = correlation(I_yy, Eigen::MatrixXd::Ones(patch_size, patch_size));
    auto sI_xy = correlation(I_xy, Eigen::MatrixXd::Ones(patch_size, patch_size));
    Eigen::MatrixXd score = ((sI_xx.array() * sI_yy.array() - 2 * sI_xy.array()) -
                             kappa * (sI_xx.array() + sI_yy.array()).square())
                                .matrix();
    score = (score.array() < 0).select(0, score);
    // std::cout << score;

    auto score_cv = convet_to_cv_to_show(score);
    cv::Mat matArray[] = {img, score_cv};
    cv::Mat out;
    cv::vconcat(matArray, 2, out);
    cv::imshow("output", out);
    cv::waitKey(0);
}

int main()
{
    std::string in_data_root = "../../data/ex03/";
    SortedImageFiles image_files(in_data_root);

    // for (auto & image_path : image_files)
    // {
    //     std::cout << image_path.number() << " " << image_path.path() << std::endl;
    // }

    size_t patch_size = 9;
    double harris_kappa = 0.08;
    size_t non_maximum_suppression_radius = 9;
    size_t descriptor_radius = 9;
    double match_lambda = 4;

    auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
    harris(src_img, patch_size, harris_kappa);
    return 0;
}