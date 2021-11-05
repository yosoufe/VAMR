#pragma once
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
gaussian_vector(double sigma,
                size_t radius = 0);

Eigen::MatrixXd
gaussian_blur(const Eigen::MatrixXd &src,
              double sigma);

void calculate_DoGs(
    size_t num_scales,
    const Eigen::MatrixXd &eigen_octave_img,
    std::vector<Eigen::MatrixXd> &DoGs_out,
    double sigma_zero);

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
                       size_t num_scales,
                       double sigma_zero);

std::vector<std::vector<Eigen::MatrixXd>>
compute_DoGs(std::vector<std::vector<Eigen::MatrixXd>> blurred_images);

std::vector<MatrixXS>
extract_keypoints(const std::vector<std::vector<Eigen::MatrixXd>> &DoGs,
                  double contrast_threshold);

void show_kpts_in_images(const std::vector<MatrixXS> &kpts,
                         const cv::Mat &img,
                         int num_scales_in_octave);

Eigen::MatrixXd
compute_descriptors(const std::vector<Eigen::MatrixXd> &blurred_images,
                    const MatrixXS &kpts,
                    bool rot_invariant,
                    MatrixXS &final_locations);