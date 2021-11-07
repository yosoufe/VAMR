#include "sift.hpp"

std::vector<double>
sift_sigmas(size_t num_scales, double sigma_0)
{
    std::vector<double> res;
    for (int scale = -1; scale <= int(num_scales + 1); scale++)
    {
        res.push_back(sigma_0 * std::pow(2.0, double(scale) / double(num_scales)));
    }
    return res;
}

Eigen::MatrixXd
gaussian_vector(double sigma, size_t radius)
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

Eigen::MatrixXd
gaussian_blur(const Eigen::MatrixXd &src, double sigma)
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

std::vector<Eigen::MatrixXd>
compute_image_pyramid(const cv::Mat &img,
                      size_t num_octaves)
{
    std::vector<Eigen::MatrixXd> pyramid;

    for (size_t octave = 0; octave < num_octaves; octave++)
    {
        cv::Mat octave_img;
        double scale = 1.0 / std::pow(2, octave);
        cv::resize(img, octave_img, cv::Size(), scale, scale, cv::INTER_CUBIC);
        Eigen::MatrixXd eigen_octave_img = cv_2_eigen(octave_img);
        pyramid.push_back(eigen_octave_img);
    }
    return pyramid;
}

std::vector<std::vector<Eigen::MatrixXd>>
compute_blurred_images(const std::vector<Eigen::MatrixXd> &image_pyramid,
                       size_t num_scales,
                       double sigma_zero)
{
    size_t num_octaves = image_pyramid.size();
    std::vector<std::vector<Eigen::MatrixXd>> blurred_images;
    std::vector<double> sigmas = sift_sigmas(num_scales, sigma_zero);

    for (size_t octave = 0; octave < num_octaves; octave++)
    {
        std::vector<Eigen::MatrixXd> blurred_images_in_octave;
        Eigen::MatrixXd eigen_octave_img = image_pyramid[octave];
        for (int scale = 0; scale < int(num_scales + 3); scale++)
        {
            Eigen::MatrixXd blured_img = gaussian_blur(eigen_octave_img, sigmas[scale]);
            blurred_images_in_octave.push_back(blured_img);
        }
        blurred_images.push_back(blurred_images_in_octave);
    }
    return blurred_images;
}

std::vector<std::vector<Eigen::MatrixXd>>
compute_DoGs(std::vector<std::vector<Eigen::MatrixXd>> blurred_images)
{
    std::vector<std::vector<Eigen::MatrixXd>> DoGs;
    size_t num_octaves = blurred_images.size();
    for (size_t octave = 0; octave < num_octaves; octave++)
    {
        std::vector<Eigen::MatrixXd> blurred_images_in_octave = blurred_images[octave];
        size_t num_dogs_per_octave = blurred_images_in_octave.size() - 1;
        std::vector<Eigen::MatrixXd> octave_dogs;
        for (int idx = 0; idx < int(num_dogs_per_octave); idx++)
        {
            Eigen::MatrixXd blured_up = blurred_images_in_octave[idx];
            Eigen::MatrixXd blured_down = blurred_images_in_octave[idx + 1];
            Eigen::MatrixXd DoG = (blured_up - blured_down).cwiseAbs();
            octave_dogs.push_back(DoG);
        }
        DoGs.push_back(octave_dogs);
    }
    return DoGs;
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

/**
 * @brief calculates scale, octave and location of key points from DoGs.
 *
 * @param DoGs
 * @param contrast_threshold
 * @param num_octaves
 * @return Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> (scale, u, v)
 */
std::vector<MatrixXS>
extract_keypoints(const std::vector<std::vector<Eigen::MatrixXd>> &DoGs,
                  double contrast_threshold)
{
    std::vector<MatrixXS> keypoints;
    size_t num_octaves = DoGs.size();

    int kernel_r = 1;
    double cut_ratio = 0.05; // lazy hack to ignore the edges, we should calculate how much we would like to ignore
    int scale = 1;

    // contrast_threshold = std::floor(0.5 * contrast_threshold / num_scales_in_octave * 255);
    // std::cout << contrast_threshold << std::endl;

    for (int octave = 0; octave < num_octaves; ++octave)
    {
        std::vector<size_t> kps;
        auto &octave_dogs = DoGs[octave];
        int num_dogs_in_octave = octave_dogs.size();
        int num_scales_in_octave = num_dogs_in_octave - 2;

        int cols = octave_dogs[0].cols();
        int rows = octave_dogs[0].rows();

        for (int scale_in_octave = 0; scale_in_octave < num_scales_in_octave; ++scale, ++scale_in_octave)
        {
            int dog_idx = scale_in_octave + 1;
            // ignore close to borders
            for (int u = std::max(kernel_r, int(cut_ratio * cols));
                 u < std::min(cols - kernel_r, int((1 - cut_ratio) * cols));
                 ++u)
            {
                for (int v = std::max(kernel_r, int(cut_ratio * rows));
                     v < std::min(rows - kernel_r, int((1 - cut_ratio) * rows));
                     ++v)
                {
                    if (is_max_in_window(octave_dogs, dog_idx, u, v, contrast_threshold, kernel_r))
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
            MatrixXS res_mat = res;
            keypoints.push_back(res_mat);
        }
        else
        {
            keypoints.push_back(MatrixXS());
        }
    }

    return keypoints;
}

void show_kpts_in_images(const std::vector<MatrixXS> &kpts,
                         const cv::Mat &img,
                         int num_scales_in_octave)
{
    int num_octaves = kpts.size();

    for (int octave = 0; octave < num_octaves; ++octave)
    {
        cv::Mat octave_img;
        double scale = 1.0 / std::pow(2, octave);
        cv::resize(img, octave_img, cv::Size(), scale, scale);
        cv::cvtColor(octave_img, octave_img, cv::COLOR_GRAY2BGR);

        auto & octave_kpts = kpts[octave];

        for (size_t kpt_idx = 0 ; kpt_idx < octave_kpts.cols(); ++kpt_idx)
        {
            cv::circle(octave_img, cv::Point2d(octave_kpts(1, kpt_idx), octave_kpts(2, kpt_idx)), 3, cv::Scalar(0, 0, 255));
        }

        std::stringstream s;
        s << "octave " << octave;
        show(octave_img, s.str());
    }
}

Eigen::MatrixXd
compute_descriptors(const std::vector<Eigen::MatrixXd> &blurred_images,
                    const MatrixXS &kpts,
                    bool rot_invariant,
                    MatrixXS &final_locations)
{
    return Eigen::MatrixXd();
}
