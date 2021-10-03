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

Eigen::MatrixXd cv_2_eigen(const cv::Mat &img)
{
    cv::Mat img_double;
    Eigen::MatrixXd eigen_img;
    cv::cv2eigen(img, eigen_img);
    return eigen_img;
}

void visualize_matrix_as_image(Eigen::MatrixXd mat)
{
    auto mat_cv = convet_to_cv_to_show(mat);
    cv::imshow("output", mat_cv);
    cv::waitKey(0);
}

void viz_score_image(const Eigen::MatrixXd &score, const cv::Mat &img)
{
    auto score_cv = convet_to_cv_to_show(score);
    // std::cout << score_cv.size() << " " << img.size() << std::endl;
    // std::cout << score_cv.dims << " " << img.dims << std::endl;
    // std::cout << score_cv.type() << " " << img.type() << std::endl;
    cv::Mat matArray[] = {img, score_cv};
    cv::Mat out;
    cv::vconcat(matArray, 2, out);
    cv::imshow("output", out);
    cv::waitKey(0);
}

auto shi_tomasi(const cv::Mat &img, size_t patch_size)
{
    Eigen::MatrixXd eigen_img = cv_2_eigen(img);
    auto I_x = correlation(eigen_img, sobel_x_kernel());
    auto I_y = correlation(eigen_img, sobel_y_kernel());
    auto I_xx = I_x.array().square().matrix();
    auto I_yy = I_y.array().square().matrix();
    auto I_xy = (I_x.array() * I_y.array()).matrix();
    auto sI_xx = correlation(I_xx, Eigen::MatrixXd::Ones(patch_size, patch_size));
    auto sI_yy = correlation(I_yy, Eigen::MatrixXd::Ones(patch_size, patch_size));
    auto sI_xy = correlation(I_xy, Eigen::MatrixXd::Ones(patch_size, patch_size));

    auto trace = (sI_xx + sI_yy).array();
    auto determinant = (sI_xx.array() * sI_yy.array()) - sI_xy.array().square();
    Eigen::MatrixXd score = (trace / 2.0 - ((trace / 2.0).square() - determinant).sqrt()).matrix();
    score = (score.array() < 0).select(0, score);
    //viz_score_image(score, img);
    return score;
}

auto harris(cv::Mat &img, size_t patch_size, double kappa)
{
    Eigen::MatrixXd eigen_img = cv_2_eigen(img);

    auto I_x = correlation(eigen_img, sobel_x_kernel());
    auto I_y = correlation(eigen_img, sobel_y_kernel());
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

    // viz_score_image(score, img);
    return score;
}

void viz_harris_shitomasi_scores(Eigen::MatrixXd harris, Eigen::MatrixXd shi_tomasi, cv::Mat src_img)
{
    auto harris_cv = convet_to_cv_to_show(harris);
    auto shi_tomasi_cv = convet_to_cv_to_show(shi_tomasi);
    cv::Mat shi_merged;
    cv::vconcat(src_img, shi_tomasi_cv, shi_merged);
    cv::putText(shi_merged, "shi tomasi", cv::Point(50,shi_merged.rows * 0.95), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255),false);
    cv::Mat harris_merged;
    cv::vconcat(src_img, harris_cv, harris_merged);
    cv::putText(harris_merged, "harris", cv::Point(50,harris_merged.rows * 0.95), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255),false);
    cv::Mat res;
    cv::hconcat(shi_merged, harris_merged, res);
    cv::imshow("Image", res);
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
    auto shi_tomasi_score = shi_tomasi(src_img, patch_size);
    auto harris_score = harris(src_img, patch_size, harris_kappa);
    viz_harris_shitomasi_scores(harris_score, shi_tomasi_score, src_img);

    return 0;
}