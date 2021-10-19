#include "sift.hpp"

std::vector<double> sift_sigmas(size_t num_scales, double sigma_0)
{
    std::vector<double> res;
    for (int scale = -1; scale <= int(num_scales + 1); scale++)
    {
        res.push_back(sigma_0 * std::pow(2.0, double(scale) / double(num_scales)));
    }
    return res;
}

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

void calculate_DoGs(
    size_t num_scales,
    const Eigen::MatrixXd &eigen_octave_img,
    std::vector<Eigen::MatrixXd> &DoGs_out)
{
    Eigen::MatrixXd blured_down;
    std::vector<double> sigmas = sift_sigmas(num_scales, 1.6);

    for (int scale = -1; scale < int(num_scales + 1); scale++)
    {
        if (blured_down.size() == 0)
        {
            blured_down = gaussian_blur(eigen_octave_img, sigmas[scale + 1]);
        }
        Eigen::MatrixXd blured_up = gaussian_blur(eigen_octave_img, sigmas[scale + 2]);
        Eigen::MatrixXd DoG = (blured_down - blured_up).cwiseAbs();
        DoGs_out.push_back(DoG);
        blured_down = blured_up;
    }
}

bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      const int &scale,
                      const int &u,
                      const int &v,
                      double contrast_threshold,
                      size_t radius)
{
    bool res = true;
    double center_val = DoGs[scale](v, u);
    if (center_val < contrast_threshold)
        return false;

    for (int s = scale-1; s <= scale + 1; s++)
    {
        for (int ui = u - radius; ui <= u + radius; ui++)
        {
            for (int vi = v - radius; vi <= v + radius; vi++)
            {
                if (vi == v && ui == u) continue;
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

    std::vector<size_t> kps;
    size_t kernel_r = 3;
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
                if (is_max_in_window(DoGs, scale+1, u, v, contrast_threshold, kernel_r))
                {
                    kps.push_back(scale + 1);
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