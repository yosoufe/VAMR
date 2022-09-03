#include "keypoint_tracking.hpp"
#include "utils.hpp"
#include "operations.hpp"

void viz_score_image(const Eigen::MatrixXd &score, const cv::Mat &img)
{
    auto score_cv = convert_to_cv_to_show(score);
    cv::Mat matArray[] = {img, score_cv};
    cv::Mat out;
    cv::vconcat(matArray, 2, out);
    cv::imshow("output", out);
    cv::waitKey(0);
}

Eigen::MatrixXd select_keypoints(
    const Eigen::MatrixXd &score,
    size_t num,
    size_t radius)
{
    Eigen::MatrixXd res(2, num);
    Eigen::MatrixXd temp_score = Eigen::MatrixXd::Zero(score.rows() + 2 * radius,
                                                       score.cols() + 2 * radius);
    temp_score.block(radius, radius, score.rows(), score.cols()) = score;
    for (size_t i = 0; i < num; i++)
    {
        Eigen::MatrixXd::Index maxRow, maxCol;
        temp_score.maxCoeff(&maxRow, &maxCol);
        res(1, i) = maxRow - radius;
        res(0, i) = maxCol - radius;
        temp_score.block(
            maxRow - radius,
            maxCol - radius,
            2 * radius + 1,
            2 * radius + 1) =
            Eigen::MatrixXd::Zero(
                2 * radius + 1,
                2 * radius + 1);
    }
    return res;
}

Eigen::MatrixXd describe_keypoints(
    const Eigen::MatrixXd &img,
    Eigen::MatrixXd keypoints,
    size_t descriptor_radius)
{
    size_t descriptor_dia = 2 * descriptor_radius + 1;
    size_t descriptor_len = descriptor_dia * descriptor_dia;
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(descriptor_len, keypoints.cols());
    Eigen::MatrixXd padded_img = Eigen::MatrixXd::Zero(img.rows() + 2 * descriptor_radius, img.cols() + 2 * descriptor_radius);
    padded_img.block(descriptor_radius, descriptor_radius, img.rows(), img.cols()) = img; // eigen_img; eigen_img.cast<double>()
    for (size_t idx = 0; idx < keypoints.cols(); idx++)
    {
        size_t row = keypoints(1, idx) + descriptor_radius, col = keypoints(0, idx) + descriptor_radius;
        Eigen::MatrixXd patch = padded_img.block(row - descriptor_radius,
                                                 col - descriptor_radius,
                                                 descriptor_dia,
                                                 descriptor_dia);
        patch.resize(descriptor_len, 1);
        res.block(0, idx, descriptor_len, 1) = patch;
    }
    return res;
}

VectorXI index_of_uniques(const VectorXI &src)
{
    std::unordered_map<size_t, size_t> mappp;
    VectorXI uniques = VectorXI::Ones(src.size());
    for (size_t idx = 0; idx < src.size(); idx++)
    {
        if (mappp.find(src(idx)) == mappp.end())
        {
            mappp[src(idx)] = idx;
        }
        else
        {
            uniques(mappp[src(idx)]) = 0;
            uniques(src(idx)) = 0;
        }
    }
    return uniques;
}

VectorXI match_descriptors(
    const Eigen::MatrixXd &query_descriptors,
    const Eigen::MatrixXd &database_descriptors,
    double match_lambda)
{

    Eigen::VectorXd dists(query_descriptors.cols());
    VectorXI matches(query_descriptors.cols());

    for (size_t idx = 0; idx < query_descriptors.cols(); idx++)
    {
        // dist is 1x200
        Eigen::MatrixXd diff = database_descriptors.colwise() - query_descriptors.col(idx);
        Eigen::VectorXd dist = diff.colwise().norm();

        Eigen::MatrixXd::Index closest_kp_idx = 0;
        dists(idx) = dist.minCoeff(&closest_kp_idx);
        matches(idx) = (size_t)(closest_kp_idx);
    }

    double big_double = std::numeric_limits<double>::max();
    auto temp_score = (dists.array() == 0).select(big_double, dists);
    double min_non_zero_dist = temp_score.minCoeff();

    matches = (dists.array() >= (match_lambda * min_non_zero_dist)).select(KPTS_NO_MATCH, matches);
    auto idx_uniques = index_of_uniques(matches);
    matches = (idx_uniques.array() == 1).select(matches, KPTS_NO_MATCH);

    return matches;
}

void calculate_Is(
    const Eigen::MatrixXd &img,
    size_t patch_size,
    Eigen::MatrixXd &sI_xx,
    Eigen::MatrixXd &sI_yy,
    Eigen::MatrixXd &sI_xy)
{
    auto I_x = correlation(img, sobel_x_kernel());
    auto I_y = correlation(img, sobel_y_kernel());
    auto I_xx = I_x.array().square().matrix();
    auto I_yy = I_y.array().square().matrix();
    auto I_xy = (I_x.array() * I_y.array()).matrix();
    sI_xx = correlation(I_xx, Eigen::MatrixXd::Ones(patch_size, patch_size));
    sI_yy = correlation(I_yy, Eigen::MatrixXd::Ones(patch_size, patch_size));
    sI_xy = correlation(I_xy, Eigen::MatrixXd::Ones(patch_size, patch_size));
}

Eigen::MatrixXd shi_tomasi(const Eigen::MatrixXd &img, size_t patch_size)
{
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;
    calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    auto trace = (sI_xx + sI_yy).array();
    auto determinant = (sI_xx.array() * sI_yy.array()) - sI_xy.array().square();
    Eigen::MatrixXd score = (trace / 2.0 - ((trace / 2.0).square() - determinant).sqrt()).matrix();
    Eigen::MatrixXd shi_tomasi_score = (score.array() < 0).select(0, score);
    return shi_tomasi_score;
}

Eigen::MatrixXd harris(const Eigen::MatrixXd &img, size_t patch_size, double kappa)
{
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;
    calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    Eigen::MatrixXd score = ((sI_xx.array() * sI_yy.array() - 2 * sI_xy.array()) -
                             kappa * (sI_xx.array() + sI_yy.array()).square())
                                .matrix();
    Eigen::MatrixXd harris_score = (score.array() < 0).select(0, score);
    return harris_score;
}

Eigen::MatrixXd non_maximum_suppression(
    const Eigen::MatrixXd &input,
    size_t patch_size)
{
    int radius = patch_size / 2;
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(input.rows(), input.cols());

    for (int center_row = 0; center_row < input.rows(); ++center_row)
    {
        for (int center_col = 0; center_col < input.cols(); ++center_col)
        {
            auto center = input(center_row, center_col);
            bool suppress = false;
            for(int comp_row = center_row - radius; comp_row <= center_row + radius ; ++comp_row)
            {
                if (comp_row < 0 || comp_row >= input.rows())
                    continue;
                
                for(int comp_col = center_col - radius; comp_col <= center_col + radius ; ++comp_col)
                {
                    if (comp_col < 0 || comp_col >= input.cols())
                        continue;
                    
                    auto compare = input(comp_row, comp_col);

                    if (compare > center)
                    {
                        suppress = true;
                        break;
                    }
                }
                if (suppress)
                    break;
            }
            if (!suppress)
                output(center_row,center_col) = input(center_row, center_col);
        }
    }
    return output;
}

cv::Mat viz_harris_shi_tomasi_scores(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    bool show_img)
{
    // std::cout << std::endl << shi_tomasi_score.block(0,0, 10,10) << std::endl; // TODO
    // std::cout << std::endl << harris_score.block(0,0, 10,10) << std::endl;
    cv::Mat src_img = img.clone();
    auto harris_cv = convert_to_cv_to_show(harris_score);
    auto shi_tomasi_cv = convert_to_cv_to_show(shi_tomasi_score);
    cv::Mat shi_merged;
    cv::vconcat(src_img, shi_tomasi_cv, shi_merged);
    cv::putText(shi_merged,
                "shi tomasi score",
                cv::Point(50, shi_merged.rows * 0.95),
                cv::FONT_HERSHEY_DUPLEX,
                1,
                cv::Scalar(255),
                false);
    cv::Mat harris_merged;
    cv::vconcat(src_img, harris_cv, harris_merged);
    cv::putText(harris_merged,
                "harris score",
                cv::Point(50, harris_merged.rows * 0.95),
                cv::FONT_HERSHEY_DUPLEX,
                1,
                cv::Scalar(255),
                false);
    cv::Mat res;
    cv::hconcat(shi_merged, harris_merged, res);
    if (show_img)
    {
        cv::imshow("Image", res);
        cv::waitKey(0);
    }

    return res;
}

void add_keypoints(cv::Mat &src, const Eigen::MatrixXd &keypoints)
{
    for (size_t idx = 0; idx < keypoints.cols(); idx++)
    {
        cv::drawMarker(
            src,
            cv::Point(keypoints(0, idx), keypoints(1, idx)),
            cv::Scalar(0, 0, 255),
            cv::MARKER_TILTED_CROSS,
            10,
            2);
    }
}

cv::Mat viz_key_points(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    const Eigen::MatrixXd &harris_kps,
    const Eigen::MatrixXd &shi_tomasi_kps,
    bool show_img)
{
    cv::Mat score_img = viz_harris_shi_tomasi_scores(img,
                                                     shi_tomasi_score,
                                                     harris_score,
                                                     false);
    cv::Mat color_src;
    cv::cvtColor(img, color_src, cv::COLOR_GRAY2BGR);
    cv::Mat harris_kps_img = color_src.clone();
    cv::Mat shi_tomasi_kps_img = color_src.clone();
    add_keypoints(harris_kps_img, harris_kps);
    add_keypoints(shi_tomasi_kps_img, shi_tomasi_kps);
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

cv::Mat viz_dscr(
    const cv::Mat &img,
    const Eigen::MatrixXd desc)
{
    size_t cols = img.cols;
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

cv::Mat viz_descriptors(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    const Eigen::MatrixXd &harris_kps,
    const Eigen::MatrixXd &shi_tomasi_kps,
    const Eigen::MatrixXd &harris_descriptors,
    const Eigen::MatrixXd &shi_tomasi_descriptors,
    bool show_img)
{
    cv::Mat key_pts_img = viz_key_points(img, shi_tomasi_score, harris_score, shi_tomasi_kps, harris_kps, false);
    cv::Mat desc_harris = viz_dscr(img, harris_descriptors);
    cv::Mat desc_shi_tomasi = viz_dscr(img, shi_tomasi_descriptors);
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

cv::Mat viz_matches(const cv::Mat &src_img,
                    const VectorXI &matches,
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
        if (matches(idx) == KPTS_NO_MATCH)
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