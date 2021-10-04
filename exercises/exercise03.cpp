#include "utils.hpp"
#include "folder_manager.hpp"
#include <cassert>

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

class ShiTomasAndHarris
{
private:
    cv::Mat src_img;
    Eigen::MatrixXd eigen_img, m_harris_score, m_shi_tomasi_score;
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;
    size_t patch_size;
    double harris_kappa;
    Eigen::MatrixXd m_harris_kps, m_shi_tomasi_kps;

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

    void calculate_Is()
    {
        if (sI_xx.size() == 0)
        {
            eigen_img = cv_2_eigen(src_img);
            auto I_x = correlation(eigen_img, sobel_x_kernel());
            auto I_y = correlation(eigen_img, sobel_y_kernel());
            auto I_xx = I_x.array().square().matrix();
            auto I_yy = I_y.array().square().matrix();
            auto I_xy = (I_x.array() * I_y.array()).matrix();
            sI_xx = correlation(I_xx, Eigen::MatrixXd::Ones(patch_size, patch_size));
            sI_yy = correlation(I_yy, Eigen::MatrixXd::Ones(patch_size, patch_size));
            sI_xy = correlation(I_xy, Eigen::MatrixXd::Ones(patch_size, patch_size));
        }
    }

    Eigen::MatrixXd select_kps(const Eigen::MatrixXd &scores, size_t num, size_t radius)
    {
        Eigen::MatrixXd res(2, num);
        Eigen::MatrixXd temp_score = Eigen::MatrixXd::Zero(scores.rows() + 2 * radius,
                                                           scores.cols() + 2 * radius);
        temp_score.block(radius, radius, scores.rows(), scores.cols()) = scores;
        for (size_t i = 0; i < num; i++)
        {
            Eigen::MatrixXd::Index maxRow, maxCol;
            temp_score.maxCoeff(&maxRow, &maxCol);
            res(1, i) = maxRow - radius;
            res(0, i) = maxCol - radius;

            temp_score.block(maxRow - radius, maxCol - radius, 2 * radius + 1, 2 * radius + 1) = Eigen::MatrixXd::Zero(2 * radius + 1, 2 * radius + 1);
        }
        return res;
    }

    void add_keypoints(cv::Mat &src, const Eigen::MatrixXd &keypoints) const
    {
        for (size_t idx= 0 ; idx< keypoints.cols(); idx++)
        {
            cv::drawMarker(src, cv::Point(keypoints(0,idx), keypoints(1,idx)), cv::Scalar(0,0,255),cv::MARKER_TILTED_CROSS, 10, 2);
        }
    }

public:
    ShiTomasAndHarris(const cv::Mat &img,
                      size_t patch_size,
                      double harris_kappa) : src_img(img.clone()),
                                             patch_size(patch_size),
                                             harris_kappa(harris_kappa) {}

    Eigen::MatrixXd harris_score()
    {
        if (m_harris_score.size() != 0)
            return m_harris_score;
        calculate_Is();
        Eigen::MatrixXd score = ((sI_xx.array() * sI_yy.array() - 2 * sI_xy.array()) -
                                 harris_kappa * (sI_xx.array() + sI_yy.array()).square())
                                    .matrix();
        m_harris_score = (score.array() < 0).select(0, score);
        // viz_score_image(score, img);
        return m_harris_score;
    }

    Eigen::MatrixXd shi_tomasi_score()
    {
        if (m_shi_tomasi_score.size() != 0)
            return m_shi_tomasi_score;
        calculate_Is();
        auto trace = (sI_xx + sI_yy).array();
        auto determinant = (sI_xx.array() * sI_yy.array()) - sI_xy.array().square();
        Eigen::MatrixXd score = (trace / 2.0 - ((trace / 2.0).square() - determinant).sqrt()).matrix();
        m_shi_tomasi_score = (score.array() < 0).select(0, score);
        //viz_score_image(score, img);
        return m_shi_tomasi_score;
    }

    cv::Mat viz_harris_shitomasi_scores(bool show_img = true)
    {
        harris_score();
        shi_tomasi_score();
        auto harris_cv = convet_to_cv_to_show(m_harris_score);
        auto shi_tomasi_cv = convet_to_cv_to_show(m_shi_tomasi_score);
        cv::Mat shi_merged;
        cv::vconcat(src_img, shi_tomasi_cv, shi_merged);
        cv::putText(shi_merged, "shi tomasi score", cv::Point(50, shi_merged.rows * 0.95), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255), false);
        cv::Mat harris_merged;
        cv::vconcat(src_img, harris_cv, harris_merged);
        cv::putText(harris_merged, "harris score", cv::Point(50, harris_merged.rows * 0.95), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255), false);
        cv::Mat res;
        cv::hconcat(shi_merged, harris_merged, res);
        if (show_img)
        {
            cv::imshow("Image", res);
            cv::waitKey(0);
        }

        return res;
    }

    Eigen::MatrixXd select_harris_keypoints(size_t num, size_t radius)
    {
        if (m_harris_kps.size() != 0)
            return m_harris_kps;

        harris_score();
        m_harris_kps = select_kps(m_harris_score, num, radius);
        return m_harris_kps;
    }

    Eigen::MatrixXd select_shi_tomasi_keypoints(size_t num, size_t radius)
    {
        if (m_shi_tomasi_kps.size() != 0)
            return m_shi_tomasi_kps;

        shi_tomasi_score();
        m_shi_tomasi_kps = select_kps(m_shi_tomasi_score, num, radius);
        return m_shi_tomasi_kps;
    }

    cv::Mat viz_key_points(bool show_img = true)
    {
        if (m_harris_kps.size()==0 || m_shi_tomasi_kps.size() == 0)
        {
            throw std::runtime_error("first call 'select_harris_keypoints' and 'select_shi_tomasi_keypoints' functions");
        }
        cv::Mat score_img = viz_harris_shitomasi_scores(false);
        cv::Mat color_src;
        cv::cvtColor (src_img, color_src, cv::COLOR_GRAY2BGR);
        cv::Mat harris_kps_img = color_src.clone();
        cv::Mat shi_tomasi_kps_img = color_src.clone();
        add_keypoints(harris_kps_img, m_harris_kps);
        add_keypoints(shi_tomasi_kps_img, m_shi_tomasi_kps);
        cv::Mat temp;
        cv::hconcat(shi_tomasi_kps_img, harris_kps_img, temp);
        cv::Mat res;
        cv::cvtColor (score_img, score_img, cv::COLOR_GRAY2BGR);
        cv::vconcat(score_img, temp, res);
        if (show_img)
        {
            cv::imshow("Image", res);
            cv::waitKey(0);
        }
        return res;
    }
};

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
    size_t num_keypoints = 200;
    size_t descriptor_radius = 9;
    double match_lambda = 4;

    // Part 1: calculate corner response functions
    auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
    ShiTomasAndHarris tracker(src_img, patch_size, harris_kappa);
    auto shi_tomasi_score = tracker.shi_tomasi_score();
    auto harris_score = tracker.harris_score();
    // tracker.viz_harris_shitomasi_scores();

    // Part 2: Select keypoints
    auto harris_kps = tracker.select_harris_keypoints(num_keypoints, non_maximum_suppression_radius);
    auto shi_tomasi_kps = tracker.select_shi_tomasi_keypoints(num_keypoints, non_maximum_suppression_radius);
    tracker.viz_key_points();
    return 0;
}