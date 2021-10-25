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

Eigen::MatrixXd gaussian_vector(double sigma, size_t radius)
{
    if (radius == 0)
    {
        // from lecture:
        // radius = std::ceil(3.0 * sigma);

        // from opencv:
        // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/smooth.dispatch.cpp#L288
        radius = std::ceil(4.0 * sigma);
    }
    Eigen::MatrixXd kernel(2 * radius + 1, 1);
    for (int x = 0; x <= radius; x++)
    {
        double exponent = (-1.0) * ((x * x) / (2.0 * sigma * sigma));
        double val = std::exp(exponent);
        kernel(radius + x, 0) = val;
        kernel(radius - x, 0) = val;
    }
    kernel = kernel / kernel.sum();
    return kernel;
}

Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma)
{
    // Just to confirm:
    // https://github.com/mikepound/convolve/blob/master/run.gaussian.py
    auto kernel = gaussian_vector(sigma);
    Eigen::MatrixXd kernel_transpose = kernel.transpose();
    Eigen::MatrixXd temp = correlation(src, kernel);
    Eigen::MatrixXd blured = correlation(temp, kernel_transpose);
    // std::cout << "sigma: " << sigma << std::endl;
    // show(convet_to_cv_to_show(blured));
    // std::cout << "shown" << std::endl;
    return blured;
}

void calculate_DoGs(
    size_t num_scales,
    const Eigen::MatrixXd &eigen_octave_img,
    std::vector<Eigen::MatrixXd> &DoGs_out,
    double sigma_zero)
{
    Eigen::MatrixXd blured_down;
    std::vector<double> sigmas = sift_sigmas(num_scales, sigma_zero);

    for (int scale = -1; scale < int(num_scales + 1); scale++)
    {
        if (blured_down.size() == 0)
        {
            blured_down = gaussian_blur(eigen_octave_img, sigmas[scale + 1]);
        }
        Eigen::MatrixXd blured_up = gaussian_blur(eigen_octave_img, sigmas[scale + 2]);
        Eigen::MatrixXd DoG = (blured_up - blured_down).cwiseAbs();
        DoGs_out.push_back(DoG);
        blured_down = blured_up;
    }
}

bool is_max_in_window(const std::vector<Eigen::MatrixXd> &DoGs,
                      int dog_idx,
                      int u,
                      int v,
                      double contrast_threshold,
                      int radius)
{
    double center_val = DoGs[dog_idx](v, u);
    if (center_val <= contrast_threshold)
        return false;

    for (int s = dog_idx - 1; s <= dog_idx + 1; ++s)
    {
        for (int ui = u - radius; ui <= u + radius; ++ui)
        {
            for (int vi = v - radius; vi <= v + radius; ++vi)
            {
                if (vi == v && ui == u && s == dog_idx) // don't compare with itself.
                    continue;
                if (center_val < DoGs[s](vi, ui))
                    return false;
            }
        }
    }
    return true;
}

typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXS;

/**
 * @brief calculates scale, octave and location of key points from DoGs.
 *
 * @param DoGs
 * @param contrast_threshold
 * @param num_octaves
 * @return Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> (scale, u, v)
 */
MatrixXS find_keypoints(const std::vector<Eigen::MatrixXd> &DoGs,
                        double contrast_threshold,
                        size_t num_octaves)
{
    int num_scales_in_octave = DoGs.size() / num_octaves - 2;
    int num_dogs_in_octave = DoGs.size() / num_octaves;

    std::vector<size_t> kps;
    int kernel_r = 1;
    double cut_ratio = 0.05; // lazy hack to ignore the edges, we should calculate how much we would like to ignore
    int scale = 1;

    // contrast_threshold = std::floor(0.5 * contrast_threshold / num_scales_in_octave * 255);
    // std::cout << contrast_threshold << std::endl;

    for (int octave = 0; octave < num_octaves; ++octave)
    {
        int cols = DoGs[num_dogs_in_octave * octave].cols();
        int rows = DoGs[num_dogs_in_octave * octave].rows();

        for (int scale_in_octave = 0; scale_in_octave < num_scales_in_octave; ++scale, ++scale_in_octave)
        {
            int dog_idx = num_dogs_in_octave * octave + scale_in_octave + 1;
            // ignore close to borders
            for (int u = std::max(kernel_r, int(cut_ratio * cols));
                 u < std::min(cols - kernel_r, int((1 - cut_ratio) * cols));
                 ++u)
            {
                for (int v = std::max(kernel_r, int(cut_ratio * rows));
                     v < std::min(rows - kernel_r, int((1 - cut_ratio) * rows));
                     ++v)
                {
                    if (is_max_in_window(DoGs, dog_idx, u, v, contrast_threshold, kernel_r))
                    {
                        kps.push_back(scale);
                        kps.push_back(octave);
                        kps.push_back(u);
                        kps.push_back(v);
                    }
                }
            }
        }
    }

    if (kps.size())
    {
        size_t *ptr = &kps[0];
        Eigen::Map<MatrixXS> res(ptr, 4, kps.size() / 4);
        size_t num_kps = kps.size() / 4;
        res.resize(4, num_kps);
        return res;
    }
    return MatrixXS();
}

void show_kpts_in_images(const MatrixXS &kpts,
                         const cv::Mat &img,
                         int num_octaves,
                         int num_scales_in_octave)
{
    size_t num_kpts = kpts.cols();
    size_t kpts_idx = 0;
    int scl = 1;

    for (int octave = 0; octave < num_octaves; ++octave)
    {
        // for (int scale_in_octave = 0; scale_in_octave < num_scales_in_octave; ++scale_in_octave, ++scl)
        // {
        cv::Mat octave_img;
        double scale = 1.0 / std::pow(2, octave);
        cv::resize(img, octave_img, cv::Size(), scale, scale);
        cv::cvtColor(octave_img, octave_img, cv::COLOR_GRAY2BGR);

        for (; kpts_idx < kpts.cols() && kpts(1, kpts_idx) == octave; ++kpts_idx) //  && kpts(0, kpts_idx) == scl
        {
            cv::circle(octave_img, cv::Point2d(kpts(2, kpts_idx), kpts(3, kpts_idx)), 3, cv::Scalar(0, 0, 255));
        }

        std::stringstream s;
        s << "octave " << octave << " scale " << scl;
        show(octave_img, s.str());
        // }
    }
}