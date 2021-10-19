#include "sift.hpp"

Eigen::MatrixXd gaussian_vector(double sigma)
{
    size_t radius = std::ceil(3.0 * sigma);
    Eigen::MatrixXd kernel(2 * radius + 1, 1);
    for (int x = 0; x <= radius; x++)
    {
        double cst = 1.0 / (2.0 * M_PI * sigma * sigma);
        double exponent = (-1.0) * ((x * x) / (2 * sigma * sigma));
        double val = cst * std::exp(exponent);
        kernel(radius + x, 0) = val;
        kernel(radius - x, 0) = val;
    }
    kernel = kernel / kernel.sum();
    return kernel;
}

Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma)
{
    auto kernel = gaussian_vector(sigma);
    Eigen::MatrixXd temp = correlation(src, kernel);
    Eigen::MatrixXd blured = correlation(temp, kernel.transpose());
    return blured;
}

bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      const int &scale,
                      const int &u,
                      const int &v,
                      double contrast_threshold)
{
    bool res = true;
    double center_val = DoGs[scale + 1](u, v);
    if (center_val < contrast_threshold)
        return false;
    for (int s = scale; s <= scale + 2; s++)
    {
        for (int ui = u - 1; ui <= u + 1; ui++)
        {
            for (int vi = v - 1; vi <= v + 1; vi++)
            {
                res = res && (center_val > DoGs[s](vi, ui));
                if (!res)
                    return res;
            }
        }
    }
    return res;
}

typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXS;

/**
 * @brief calculates kye points locations and scale from DoGs
 *
 * @param DoGs
 * @return Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> (scale, u, v)
 */
MatrixXS find_keypoints(const std::vector<Eigen::MatrixXd> &DoGs,
                        double contrast_threshold)
{
    int num_scales = DoGs.size() - 2;
    int cols = DoGs[0].cols();
    int rows = DoGs[0].rows();
    std::vector<size_t> res_vector;

    std::vector<size_t> kps;
    size_t kernel_r = 1;
    for (int scale = 0; scale < num_scales; scale++)
    {
        // ignore close to borders
        for (int u = std::max(kernel_r, size_t(0.1 * cols));
             u < std::min(cols - kernel_r, size_t(0.9 * cols));
             u++)
        {
            for (int v = std::max(kernel_r, size_t(0.1 * rows));
                 v < std::min(rows - kernel_r, size_t(0.9 * rows));
                 v++)
            {
                if (is_max_in_window(DoGs, scale, u, v, contrast_threshold))
                {
                    kps.push_back(scale);
                    kps.push_back(u);
                    kps.push_back(v);
                }
            }
        }
    }

    if (kps.size())
    {
        size_t *ptr = &kps[0];
        Eigen::Map<MatrixXS> res(ptr, 3, kps.size() / 3);
        size_t num_kps = kps.size() / 3;
        res.resize(3, num_kps);
        return res;
    }
    return MatrixXS();
}