#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "utils.hpp"
#include "cuda_types.hpp"

#define KPTS_NO_MATCH -1 // when there is no match for a keypoints in the matching process

using VectorXI = Eigen::Matrix<int, Eigen::Dynamic, 1>;

void viz_score_image(
    const Eigen::MatrixXd &score,
    const cv::Mat &img);

void calculate_Is(
    const Eigen::MatrixXd &img,
    size_t patch_size,
    Eigen::MatrixXd &sI_xx,
    Eigen::MatrixXd &sI_yy,
    Eigen::MatrixXd &sI_xy);

/**
 * @brief calculates harris score
 *
 * @param img the input image as Eigen Matrix
 * @param patch_size
 * @param kappa
 * @return Eigen::MatrixXd the score matrix, same size as input image.
 */
Eigen::MatrixXd harris(const Eigen::MatrixXd &img, size_t patch_size, double kappa);

/**
 * @brief calculates shi tomasi score
 *
 * @param img the input image as Eigen Matrix
 * @param patch_size
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd shi_tomasi(const Eigen::MatrixXd &img, size_t patch_size);

Eigen::MatrixXd non_maximum_suppression(const Eigen::MatrixXd &img, size_t patch_size);

/**
 * @brief Select keypoints from the score image.
 * The non-maximum suppression is applied with a box of
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
 * @return VectorXI
 */
VectorXI match_descriptors(
    const Eigen::MatrixXd &query_descriptors,
    const Eigen::MatrixXd &database_descriptors,
    double match_lambda);

cv::Mat viz_harris_shi_tomasi_scores(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    bool show_img = true);

cv::Mat viz_key_points(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    const Eigen::MatrixXd &harris_kps,
    const Eigen::MatrixXd &shi_tomasi_kps,
    bool show_img = true);

cv::Mat viz_descriptors(
    const cv::Mat &img,
    const Eigen::MatrixXd &shi_tomasi_score,
    const Eigen::MatrixXd &harris_score,
    const Eigen::MatrixXd &harris_kps,
    const Eigen::MatrixXd &shi_tomasi_kps,
    const Eigen::MatrixXd &harris_descriptors,
    const Eigen::MatrixXd &shi_tomasi_descriptors,
    bool show_img = true);

cv::Mat viz_matches(const cv::Mat &src_img,
                    const VectorXI &matches,
                    const Eigen::MatrixXd &curr_kps,
                    const Eigen::MatrixXd &prev_kps);

#if WITH_CUDA

namespace cuda
{
    void calculate_Is(
        const CuMatrixD &img,
        size_t patch_size,
        CuMatrixD &sI_xx,
        CuMatrixD &sI_yy,
        CuMatrixD &sI_xy);

    /**
     * @brief calculates harris score
     *
     * @param img the input image as CuMatrixD
     * @param patch_size
     * @param kappa
     * @return Eigen::MatrixXd the score matrix, same size as input image.
     */
    CuMatrixD harris(const CuMatrixD &img, size_t patch_size, double kappa);

    /**
     * @brief calculates shi tomasi score
     *
     * @param img the input image as CuMatrixD
     * @param patch_size
     * @return Eigen::MatrixXd
     */
    CuMatrixD shi_tomasi(const CuMatrixD &img, size_t patch_size);

    /**
     * @brief applies non_maximum suppression to the score matrix
     *
     * In each patch, if the center is the max, keep the values, otherwise
     * assigns zeros to the center.
     *
     * This is using global memory in GPU.
     *
     * @param input         The input matrix
     * @param patch_size    patch size
     * @return CuMatrixD    The output result
     */
    CuMatrixD non_maximum_suppression_1(const CuMatrixD &input, size_t patch_size);

    /**
     * @brief applies non_maximum suppression to the score matrix
     *
     * In each patch, if the center is the max, keep the values, otherwise
     * assigns zeros to the center.
     *
     * This is using shared memory in GPU.
     *
     * @param input         The input matrix
     * @param patch_size    patch size
     * @return CuMatrixD    The output result
     */
    CuMatrixD non_maximum_suppression_2(const CuMatrixD &input, size_t patch_size);

    CuMatrixD non_maximum_suppression_3(const CuMatrixD &input, size_t patch_size);

    /**
     * @brief sort the matrix in place and returns the sorted one.
     *
     * @param input             The input matrix, it will be sorted inplace.
     * @param indicies_output   The indicies before sort (2 x n_elements()).
     * @return CuMatrixD        The same input after sort. It points to the same location as input.
     */
    CuMatrixD sort_matrix(
        CuMatrixD &&input,
        CuMatrixD &indicies_output);

    /**
     * @brief Sort out of place
     *
     * @param input             input matrix,
     * @param indicies_output   indicies of the sorted elements in the input (2 x n_elements())
     * @return CuMatrixD        returns sorted matrix.
     */
    CuMatrixD sort_matrix(
        const CuMatrixD &input,
        CuMatrixD &indicies_output);

    /**
     * @brief Select keypoints from the score image.
     * The non-maximum suppression is applied with a box of
     * size (2 radius +1) * (2 radius + 1)
     *
     * @param score The score image
     * @param num   Number of best keypoints to select
     * @param radius  The radius for non-maximum suppression
     * @return Eigen::MatrixXd in shape of (2 X n_rows * n_cols)
     */
    cuda::CuMatrixD select_keypoints(
        const CuMatrixD &score,
        size_t radius);

    /**
     * @brief This is a bit different than cpu
     * implementation. sorted_pixels_based_on_scores is similar to 
     * keypoints in the cpu implementation, except it is list of all 
     * indicies in the image, and in the descending order for their 
     * corner score.
     * In cpu implementation, num_keypoints_to_consider is not needed since
     * the dimension of keypoints defines the number.
     *
     * @param img                           input image.
     * @param sorted_pixels_based_on_scores index of the pixels in the order of their scores in descending order
     * @param num_keypoints_to_consider     number of keypoints to consider. (not all of them).
     * @param descriptor_radius             radius of the descriptor.
     * @return cuda::CuMatrixD              (descriptor length x N) matrix of descriptors,
     *                                      descriptor length = (2r+1)^2,
     *                                      N = num_keypoints_to_consider
     */
    cuda::CuMatrixD describe_keypoints(
        const cuda::CuMatrixD &img,
        const cuda::CuMatrixD &sorted_pixels_based_on_scores,
        int num_keypoints_to_consider,
        int descriptor_radius);

    /**
     * @brief Returns a 1xQ matrix where the i-th coefficient is the index of the
     * database descriptor which matches to the i-th query descriptor.
     * The descriptor vectors are desc_size x num_kps.
     * matches(i) will be zero if there is no database descriptor
     * with an SSD < lambda * min(SSD). No two non-zero elements of matches will
     * be equal.
     *
     * @param query_descriptors (desc_size X num_query_kps)
     * @param database_descriptors (desc_size X num_database_kps)
     * @param match_lambda
     * @return CuMatrixI in shape 0f (num of query keypoints X 1)
     */
    cuda::CuMatrixI match_descriptors(
        const cuda::CuMatrixD &query_descriptors,
        const cuda::CuMatrixD &database_descriptors,
        double match_lambda);

    /**
     * @brief just for testing, don't use it outside of tests.
     */
    CuMatrixD
    test_calculate_difference_to_kps_database(
        const CuMatrixD &query,
        const CuMatrixD &database);

    /**
     * @brief just for testing, don't use it outside of tests.
     */
    CuMatrixD
    test_calculate_sum_kernel(const CuMatrixD &matrix);

    CuMatrixI
    test_find_closest_keypoints_kernel(
        const CuMatrixD &query_descriptors,
        const CuMatrixD &database_descriptors);

    /**
     * @brief just for testing, don't use it outside of tests.
     */
    int
    test_arg_min(const CuMatrixD &matrix);

}

#endif