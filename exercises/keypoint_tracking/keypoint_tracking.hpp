#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>

using VectorXuI= Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

void viz_score_image(
    const Eigen::MatrixXd &score,
    const cv::Mat &img);

/**
 * @brief calculates harris score
 * 
 * @param img the input image as Eigen Matrix
 * @param patch_size 
 * @param kappa 
 * @return Eigen::MatrixXd the score matrix, same size as input image.
 */
Eigen::MatrixXd harris(const Eigen::MatrixXd& img, size_t patch_size, double kappa);

/**
 * @brief calculates shi tomasi score
 * 
 * @param img the input image as Eigen Matrix
 * @param patch_size 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd shi_tomasi(const Eigen::MatrixXd& img, size_t patch_size);

/**
 * @brief Select keypoints from the score image.
 * The non-maximum suppression is applies with a box of
 * size (2 radius +1) * (2 radius + 1)
 *
 * @param score The score image
 * @param num   Number of best keypoints to select
 * @param radius  The radius for non-maximum suppression
 * @return Eigen::MatrixXd in shape of (2 x num)
 */
Eigen::MatrixXd select_keypoints(
    const Eigen::MatrixXd &score,
    size_t num,
    size_t radius);

/**
 * @brief Returns a (2r+1)^2xN matrix of image 
 * patch vectors based on image img and a 2xN matrix 
 * containing the keypoint coordinates.
 *
 * @param img       source image to get descriptors from
 * @param keypoints location of the keypoints in image. (2 x n)
 * @param descriptor_radius is the patch "radius".
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd describe_keypoints(
    const Eigen::MatrixXd &img,
    Eigen::MatrixXd keypoints,
    size_t descriptor_radius);

/**
 * @brief Returns a 1xQ matrix where the i-th coefficient is the index of the
 * database descriptor which matches to the i-th query descriptor.
 * The descriptor vectors are num_kp X desc_size and num_kp X desc_size.
 * matches(i) will be zero if there is no database descriptor
 * with an SSD < lambda * min(SSD). No two non-zero elements of matches will
 * be equal.
 * 
 * @param query_descriptors (num_kp X desc_size)
 * @param database_descriptors (num_kp X desc_size)
 * @param match_lambda 
 * @return VectorXuI 
 */
VectorXuI match_descriptors(
    const Eigen::MatrixXd &query_descriptors,
    const Eigen::MatrixXd &database_descriptors,
    double match_lambda);
