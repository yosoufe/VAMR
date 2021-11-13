#pragma once
#include <tuple>
#include <map>
#include "utils.hpp"

typedef Eigen::Matrix<size_t,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::ColMajor>
    MatrixXS;

std::vector<double>
sift_sigmas(size_t num_scales,
            double sigma_0);

Eigen::MatrixXd
gaussian_kernel(double sigma,
                size_t rows = 0,
                size_t cols = 1);

Eigen::MatrixXd
gaussian_blur(const Eigen::MatrixXd &src,
              double sigma);

bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      int dog_idx,
                      int u,
                      int v,
                      double contrast_threshold,
                      int radius);

std::vector<Eigen::MatrixXd>
compute_image_pyramid(const cv::Mat &img,
                      size_t num_octaves);

std::vector<std::vector<Eigen::MatrixXd>>
compute_blurred_images(const std::vector<Eigen::MatrixXd> &image_pyramid,
                       size_t num_scales_in_octave,
                       double sigma_zero);

std::vector<std::vector<Eigen::MatrixXd>>
compute_DoGs(std::vector<std::vector<Eigen::MatrixXd>> blurred_images);

std::vector<MatrixXS>
extract_keypoints(const std::vector<std::vector<Eigen::MatrixXd>> &DoGs,
                  double contrast_threshold);

void show_kpts_in_images(const std::vector<MatrixXS> &kpts,
                         const cv::Mat &img,
                         int num_scales_in_octave);

Eigen::MatrixXd concat_h(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
Eigen::MatrixXd concat_v(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

Eigen::VectorXd
weightedhistc(const Eigen::MatrixXd &vals,
              const Eigen::MatrixXd &weights,
              const Eigen::VectorXd &edges);

/**
 * @brief
 *
 * @param blurred_images
 * @param kpts
 * @param rot_invariant
 * @param final_locations
 * @return std::vector<Eigen::VectorXd>, vector of descriptors.
 */
std::vector<Eigen::VectorXd>
compute_descriptors(const std::vector<std::vector<Eigen::MatrixXd>> &blurred_images,
                    const std::vector<MatrixXS> &kpts,
                    bool rot_invariant,
                    size_t num_scales_in_octave,
                    std::vector<MatrixXS> &final_locations);

typedef std::map<size_t, std::tuple<size_t, double>> matches_t;

matches_t
match_features(const std::vector<std::vector<Eigen::VectorXd>> &descriptors,
               double max_ratio);

cv::Mat viz_matches(const std::vector<cv::Mat> &src_imgs,
                    const matches_t &matches,
                    const std::vector<std::vector<MatrixXS>> &keypoints_locations);