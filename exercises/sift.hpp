#include "utils.hpp"

typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXS;


Eigen::MatrixXd gaussian_vector(double sigma);
Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma);
bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      const int &scale,
                      const int &u,
                      const int &v,
                      double contrast_threshold);

MatrixXS find_keypoints(const std::vector<Eigen::MatrixXd> &DoGs,
                        double contrast_threshold);