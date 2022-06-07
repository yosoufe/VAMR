#include "keypoint_tracking.hpp"
#include "utils.hpp"

void viz_score_image(const Eigen::MatrixXd &score, const cv::Mat &img)
{
    auto score_cv = convert_to_cv_to_show(score);
    // std::cout << score_cv.size() << " " << img.size() << std::endl;
    // std::cout << score_cv.dims << " " << img.dims << std::endl;
    // std::cout << score_cv.type() << " " << img.type() << std::endl;
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
    const Eigen::MatrixXd &query_descriptors,
    const Eigen::MatrixXd &database_descriptors,
    double match_lambda)
{

    Eigen::VectorXd dists(query_descriptors.cols());
    VectorXuI matches(query_descriptors.cols());

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

    matches = (dists.array() >= (match_lambda * min_non_zero_dist)).select(0, matches);
    auto idx_uniques = index_of_uniques(matches);
    matches = (idx_uniques.array() == 1).select(matches, 0);

    return matches;
}

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

void calculate_Is(
    const Eigen::MatrixXd &img,
    size_t patch_size,
    Eigen::MatrixXd &sI_xx,
    Eigen::MatrixXd &sI_yy,
    Eigen::MatrixXd &sI_xy)
{
    // CUDA TODO
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
    // CUDA TODO
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
    // CUDA TODO
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;
    calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    Eigen::MatrixXd score = ((sI_xx.array() * sI_yy.array() - 2 * sI_xy.array()) -
                             kappa * (sI_xx.array() + sI_yy.array()).square())
                                .matrix();
    Eigen::MatrixXd harris_score = (score.array() < 0).select(0, score);
    return harris_score;
}