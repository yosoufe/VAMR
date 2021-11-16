#pragma once
#include "utils.hpp"
#include <map>
#include <utility>

typedef Eigen::Matrix<size_t,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::ColMajor>
    MatrixXS;

class ImageGradient
{
private:
    // caches
    std::map<std::tuple<size_t, size_t>, Eigen::MatrixXd> g_mags;
    std::map<std::tuple<size_t, size_t>, Eigen::MatrixXd> g_dirs;
    std::vector<std::vector<Eigen::MatrixXd>> blurred;
    Eigen::MatrixXd sobel_kernel();
    Eigen::MatrixXd sobel_x_kernel;
    Eigen::MatrixXd sobel_y_kernel;
    std::tuple<size_t, size_t> make_key(size_t octave, size_t scale);
    void calculate_grads(size_t octave, size_t scale);
public:
    ImageGradient(const std::vector<std::vector<Eigen::MatrixXd>> &blurred_images);
    Eigen::MatrixXd &get_grad_mag(size_t octave, size_t scale);
    Eigen::MatrixXd &get_grad_dir(size_t octave, size_t scale);
};