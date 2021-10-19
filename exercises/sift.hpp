#include "utils.hpp"

typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXS;

std::vector<double> sift_sigmas(size_t num_scales, double sigma_0);
Eigen::MatrixXd gaussian_vector(double sigma);
Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma);

void calculate_DoGs(
    size_t num_scales,
    const Eigen::MatrixXd &eigen_octave_img,
    std::vector<Eigen::MatrixXd> &DoGs_out);

bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      const int &scale,
                      const int &u,
                      const int &v,
                      double contrast_threshold,
                      size_t radius);

MatrixXS find_keypoints(const std::vector<Eigen::MatrixXd> &DoGs,
                        double contrast_threshold);