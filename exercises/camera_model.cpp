#include "camera_model.hpp"
#include <Eigen/SVD>

Eigen::Matrix2Xd distorted_pixel(const double k1,
                                 const double k2,
                                 const Eigen::Vector2d &principal_pt,
                                 const Eigen::Matrix2Xd &points)
{
    auto d_pts = points.colwise() - principal_pt;
    auto r_square = d_pts.colwise().squaredNorm();
    auto term_c = ((k1 * r_square).array() + k2 * (r_square.array().pow(2)) + 1.0);
    return (d_pts.array().rowwise() * term_c).matrix().colwise() + principal_pt;
}

Eigen::Matrix2Xd project_2_camera_frame(const Eigen::Matrix3d &intrinsics,
                                        const Eigen::Isometry3d &extrinsics,
                                        const Eigen::Matrix3Xd &points)
{
    Eigen::Matrix2Xd res;
    res.resize(2, points.cols());
    for (int col_idx = 0; col_idx < points.cols(); col_idx++)
    {
        auto homoG = intrinsics * extrinsics * Eigen::Vector3d(points.block(0, col_idx, 3, 1));
        res.block(0, col_idx, 2, 1) = homoG.block(0, 0, 2, 1) / homoG(2);
    }
    return res;
}

cv::Mat undistort_image(const cv::Mat &src_img,
                        double d1,
                        double d2,
                        const Eigen::Vector2d &principal_pt)
{
    cv::Mat res = src_img.clone();
    double u0 = principal_pt(0);
    double v0 = principal_pt(1);
    for (size_t v = 0; v < src_img.rows; v++)
    {
        for (size_t u = 0; u < src_img.cols; u++)
        {
            double r_2 = (u - u0) * (u - u0) + (v - v0) * (v - v0);
            double c = 1 + d1 * r_2 + d2 * r_2 * r_2;
            int u_d = c * (u - u0) + u0;
            int v_d = c * (v - v0) + v0;
            if (u_d >= 0 && u_d < src_img.cols && v_d >= 0 && v_d < src_img.rows)
            {
                auto temp = src_img.at<cv::Vec3b>(v_d, u_d);
                res.at<cv::Vec3b>(v, u) = temp;
            }
            else
            {
                res.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return res;
}