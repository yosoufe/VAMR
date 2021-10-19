#include "utils.hpp"
#include "folder_manager.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_map>

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
    Eigen::MatrixXd m_harris_descriptors, m_shi_tomasi_descriptors;
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
        for (size_t idx = 0; idx < keypoints.cols(); idx++)
        {
            cv::drawMarker(src, cv::Point(keypoints(0, idx), keypoints(1, idx)), cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 10, 2);
        }
    }

    Eigen::MatrixXd get_descriptors(Eigen::MatrixXd kps, size_t descriptor_radius)
    {
        size_t descriptor_dia = 2 * descriptor_radius + 1;
        size_t descriptor_len = descriptor_dia * descriptor_dia;
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(descriptor_len, kps.cols());
        Eigen::MatrixXd padded_img = Eigen::MatrixXd::Zero(eigen_img.rows() + 2 * descriptor_radius, eigen_img.cols() + 2 * descriptor_radius);
        padded_img.block(descriptor_radius, descriptor_radius, eigen_img.rows(), eigen_img.cols()) = eigen_img; // eigen_img; eigen_img.cast<double>()

        // std::cout << "test value " << padded_img(descriptor_radius+21, descriptor_radius+31) << " " << eigen_img(21,31) << std::endl;
        for (size_t idx = 0; idx < kps.cols(); idx++)
        {
            size_t row = kps(1, idx) + descriptor_radius, col = kps(0, idx) + descriptor_radius;
            Eigen::MatrixXd patch = padded_img.block(row - descriptor_radius,
                                                     col - descriptor_radius,
                                                     descriptor_dia,
                                                     descriptor_dia);
            // std::cout << patch.size() << " patch: " << std::endl << patch << std::endl;
            patch.resize(descriptor_len, 1);
            res.block(0, idx, descriptor_len, 1) = patch;

            // std::cout << patch.size()  << " patch resized: " << std::endl << patch << std::endl;
            // std::cout << "res: " << std::endl << res << std::endl;
        }
        // std::cout << "descs: " << std::endl << res << std::endl;
        return res;
    }

    cv::Mat viz_dscr(const Eigen::MatrixXd desc)
    {
        size_t cols = src_img.cols;
        size_t descriptor_size = std::lround(std::sqrt(desc.rows()));
        size_t descriptor_size_img = 10 * descriptor_size;
        size_t v_space = descriptor_size_img / 2;
        size_t grid_w = cols / 5;
        size_t rows = 4 * descriptor_size_img + 5 * v_space;
        cv::Mat res(rows, cols, CV_8U, cv::Scalar(0));

        for (size_t idx = 0; idx < 16; idx++)
        {
            size_t idx_col = idx % 4;
            size_t idx_row = idx / 4;
            Eigen::MatrixXd patch_eigen = desc.col(idx);
            patch_eigen.resize(descriptor_size, descriptor_size);
            cv::Mat patch = eigen_2_cv(patch_eigen);
            cv::resize(patch, patch, cv::Size(descriptor_size_img, descriptor_size_img), 0, 0, cv::INTER_NEAREST);
            size_t row_start = (descriptor_size_img + v_space) * (idx_row + 1) - descriptor_size_img;
            size_t row_end = (descriptor_size_img + v_space) * (idx_row + 1);
            size_t col_start = grid_w * (idx_col + 1) - descriptor_size_img / 2;
            size_t col_end = grid_w * (idx_col + 1) + descriptor_size_img / 2;
            patch.copyTo(
                res(cv::Range(row_start, row_end),
                    cv::Range(col_start, col_end)));
        }
        return res;
    }

public:
    ShiTomasAndHarris(const cv::Mat &img,
                      size_t patch_size,
                      double harris_kappa) : src_img(img.clone()),
                                             patch_size(patch_size),
                                             harris_kappa(harris_kappa)
    {
        eigen_img = cv_2_eigen(src_img);
    }

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
        if (m_harris_kps.size() == 0 || m_shi_tomasi_kps.size() == 0)
            throw std::runtime_error("first call 'select_harris_keypoints' and 'select_shi_tomasi_keypoints' functions");

        cv::Mat score_img = viz_harris_shitomasi_scores(false);
        cv::Mat color_src;
        cv::cvtColor(src_img, color_src, cv::COLOR_GRAY2BGR);
        cv::Mat harris_kps_img = color_src.clone();
        cv::Mat shi_tomasi_kps_img = color_src.clone();
        add_keypoints(harris_kps_img, m_harris_kps);
        add_keypoints(shi_tomasi_kps_img, m_shi_tomasi_kps);
        cv::Mat res;
        cv::hconcat(shi_tomasi_kps_img, harris_kps_img, res);
        cv::cvtColor(score_img, score_img, cv::COLOR_GRAY2BGR);
        cv::vconcat(score_img, res, res);
        if (show_img)
        {
            cv::imshow("Image", res);
            cv::waitKey(0);
        }
        return res;
    }

    Eigen::MatrixXd harris_descriptors(size_t descriptor_radius)
    {
        if (m_harris_kps.size() == 0)
            throw std::runtime_error("first call 'select_harris_keypoints' function");

        m_harris_descriptors = get_descriptors(m_harris_kps, descriptor_radius);

        return m_harris_descriptors;
    }

    Eigen::MatrixXd shi_tomasi_descriptors(size_t descriptor_radius)
    {
        if (m_shi_tomasi_kps.size() == 0)
            throw std::runtime_error("first call 'select_shi_tomasi_keypoints' function");

        m_shi_tomasi_descriptors = get_descriptors(m_shi_tomasi_kps, descriptor_radius);
        return m_shi_tomasi_descriptors;
    }

    cv::Mat viz_descriptors(bool show_img = true)
    {
        if (m_shi_tomasi_descriptors.size() == 0 || m_harris_kps.size() == 0)
            throw std::runtime_error("first call 'harris_descriptors' and 'shi_tomasi_descriptors' functions");

        cv::Mat key_pts_img = viz_key_points(false);
        cv::Mat desc_harris = viz_dscr(m_harris_descriptors);
        cv::Mat desc_shi_tomasi = viz_dscr(m_shi_tomasi_descriptors);
        cv::Mat res;
        cv::hconcat(desc_shi_tomasi, desc_harris, res);
        cv::cvtColor(res, res, cv::COLOR_GRAY2BGR);
        cv::vconcat(key_pts_img, res, res);
        if (show_img)
        {
            cv::imshow("Image", res);
            cv::waitKey(0);
        }
        return res;
    }
};

typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXuI;

VectorXuI index_of_uniques(const VectorXuI &src)
{
    std::unordered_map<size_t, size_t> mappp;
    VectorXuI uniques = VectorXuI::Ones(src.size());
    for (size_t idx = 0; idx < src.size(); idx++)
    {
        if (mappp.find(src(idx)) != mappp.end())
        {
            uniques(mappp[src(idx)]) = 0;
            uniques(src(idx)) = 0;
        }
        else
        {
            mappp[src(idx)] = idx;
        }
    }
    return uniques;
}

VectorXuI match_descriptors(
    const Eigen::MatrixXd &curr, // query_descriptors,          num_kp X desc_size
    const Eigen::MatrixXd &prev, // database_descriptors        num_kp X desc_size
    double match_lambda)
{
    // std::cout << "curr" << std::endl << curr << std::endl;
    // std::cout << "prev" << std::endl << prev << std::endl;

    Eigen::VectorXd dists(curr.cols());
    VectorXuI matches(curr.cols());

    for (size_t idx = 0; idx < curr.cols(); idx++)
    {
        // dist is 1x200
        Eigen::MatrixXd diff = prev.colwise() - curr.col(idx);
        Eigen::VectorXd dist = diff.colwise().norm();


        Eigen::MatrixXd::Index closest_kp_idx = 0;
        dists(idx) = dist.minCoeff(&closest_kp_idx);
        matches(idx) = (size_t)(closest_kp_idx);
    }

    // std::cout << "dists: " << std::endl << dists << std::endl;

    double big_double = std::numeric_limits<double>::max(); // std::numeric_limits<double>::max() 1000
    auto temp_score = (dists.array() == 0).select(big_double, dists);
    double min_non_zero_dist = temp_score.minCoeff();

    // std::cout << "min_dist: " << std::endl << min_dist << std::endl;

    // std::cout << "updating min_non_zero_dist to " << min_non_zero_dist << std::endl;

    matches = (dists.array() >= (match_lambda * min_non_zero_dist)).select(0, matches);
    auto idx_uniques = index_of_uniques(matches);
    // Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> temp(idx_uniques.size(), 2);
    // temp << idx_uniques, matches;
    // std::cout << temp << std::endl;
    // std::cout << "+++++++++++++++++++" << min_non_zero_dist << std::endl;
    // std::cout << "+++++++++++++++++++" << std::endl;
    matches = (idx_uniques.array() == 1).select(matches, 0);

    return matches;
}

cv::Mat viz_matches(const cv::Mat &src_img,
                 const VectorXuI &matches,
                 const Eigen::MatrixXd &curr_kps,
                 const Eigen::MatrixXd &prev_kps)
{
    cv::Mat color_img;
    cv::cvtColor(src_img, color_img, cv::COLOR_GRAY2BGR);
    for (size_t idx = 0; idx < matches.size(); idx++)
    {
        cv::drawMarker(color_img,
                       cv::Point(curr_kps(0, idx), curr_kps(1, idx)),
                       cv::Scalar(0, 0, 255),
                       cv::MARKER_TILTED_CROSS, 10, 2);
        if (matches(idx) == 0)
            continue;
        cv::line(color_img,
                 cv::Point(curr_kps(0, idx), curr_kps(1, idx)),
                 cv::Point(prev_kps(0, matches(idx)), prev_kps(1, matches(idx))),
                 cv::Scalar(0, 255, 0),
                 2);
    }
    cv::imshow("image", color_img);
    cv::waitKey(1);
    return color_img;
}

int main()
{
    std::string in_data_root = "../../data/ex03/";
    SortedImageFiles image_files(in_data_root);

    cv::Size img_size;

    size_t patch_size = 9;
    double harris_kappa = 0.08;
    size_t non_maximum_suppression_radius = 9;
    size_t num_keypoints = 200; // 200
    size_t descriptor_radius = 9; // 9
    double match_lambda = 4; // 4

    {
        // Part 1: calculate corner response functions
        auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
        img_size = src_img.size();
        ShiTomasAndHarris tracker(src_img, patch_size, harris_kappa);
        auto shi_tomasi_score = tracker.shi_tomasi_score();
        auto harris_score = tracker.harris_score();
        // tracker.viz_harris_shitomasi_scores();

        // Part 2: Select keypoints
        auto harris_kps = tracker.select_harris_keypoints(num_keypoints, non_maximum_suppression_radius);
        auto shi_tomasi_kps = tracker.select_shi_tomasi_keypoints(num_keypoints, non_maximum_suppression_radius);
        // tracker.viz_key_points();

        // Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
        tracker.harris_descriptors(descriptor_radius);
        tracker.shi_tomasi_descriptors(descriptor_radius);
        // tracker.viz_descriptors();
    }

    cv::VideoWriter video = create_video_writer(img_size, "ex03/keypoint_tracking.mp4");

    // Part 4 and 5 - Match descriptors between all images
    Eigen::MatrixXd prev_desc;
    Eigen::MatrixXd prev_kps;
    for (auto &image_path : image_files)
    {
        cv::Mat src_img = cv::imread(image_path.path(), cv::IMREAD_GRAYSCALE);
        ShiTomasAndHarris tracker(src_img, patch_size, harris_kappa);
        Eigen::MatrixXd curr_kps = tracker.select_harris_keypoints(num_keypoints, non_maximum_suppression_radius);
        Eigen::MatrixXd desc = tracker.harris_descriptors(descriptor_radius);
        // tracker.select_shi_tomasi_keypoints(num_keypoints, non_maximum_suppression_radius);
        // tracker.shi_tomasi_descriptors(descriptor_radius);
        // tracker.viz_descriptors();
        if (prev_desc.size() != 0)
        {
            auto matches = match_descriptors(desc, prev_desc, match_lambda);
            video << viz_matches(src_img, matches, curr_kps, prev_kps);
        //     if (image_path.number() == 2)
        //     {
        //         int temp = system("read temp");
        //     }
        }
        prev_desc = desc;
        prev_kps = curr_kps;
    }

    return 0;
}