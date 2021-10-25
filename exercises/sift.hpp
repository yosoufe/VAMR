#pragma once
#include "utils.hpp"

typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXS;

std::vector<double> sift_sigmas(size_t num_scales, double sigma_0);
Eigen::MatrixXd gaussian_vector(double sigma, size_t radius = 0);
Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma);

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

MatrixXS find_keypoints(const std::vector<Eigen::MatrixXd> &DoGs,
                        double contrast_threshold,
                        size_t num_octaves);

void show_kpts_in_images(const MatrixXS &kpts,
                         const cv::Mat &img,
                         int num_octaves,
                         int num_scales_in_octave);