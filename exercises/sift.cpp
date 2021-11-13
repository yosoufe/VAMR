#include "sift.hpp"
#include <cassert>
#include "image_gradient.hpp"

#include <chrono>
using namespace std::chrono;

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
gaussian_kernel(double sigma,
                size_t rows,
                size_t cols)
{
    if (rows == 0)
    {
        // from lecture:
        // radius = std::ceil(3.0 * sigma);
        // from opencv:
        // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/smooth.dispatch.cpp#L288
        rows = std::ceil(4.0 * sigma);
    }
    if (cols == 0)
    {
        cols = std::ceil(4.0 * sigma);
    }
    Eigen::MatrixXd kernel(rows, cols);
    double row_center = (rows + 1) / 2.0;
    double col_center = (cols + 1) / 2.0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            double dist_sq = std::pow(col_center - col, 2) + std::pow(row_center - row, 2);
            double exponent = (-1.0) * (dist_sq / (2.0 * sigma * sigma));
            double val = std::exp(exponent);
            kernel(row, col) = val;
        }
    }
    kernel = kernel / kernel.sum();
    return kernel;
}

Eigen::MatrixXd
gaussian_blur(const Eigen::MatrixXd &src, double sigma)
{
    // Just to confirm:
    // https://github.com/mikepound/convolve/blob/master/run.gaussian.py
    auto kernel = gaussian_kernel(sigma, 0, 1);
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
                       size_t num_scales_in_octave,
                       double sigma_zero)
{
    size_t num_octaves = image_pyramid.size();
    std::vector<std::vector<Eigen::MatrixXd>> blurred_images;
    std::vector<double> sigmas = sift_sigmas(num_scales_in_octave, sigma_zero);

    for (size_t octave = 0; octave < num_octaves; octave++)
    {
        std::vector<Eigen::MatrixXd> blurred_images_in_octave;
        Eigen::MatrixXd eigen_octave_img = image_pyramid[octave];
        for (int scale = 0; scale < int(num_scales_in_octave + 3); scale++)
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
    int scale = 0;

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

        for (int scale_in_octave = 0; scale_in_octave < num_scales_in_octave; ++scale_in_octave)
        {
            ++scale;
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

        auto &octave_kpts = kpts[octave];

        for (size_t kpt_idx = 0; kpt_idx < octave_kpts.cols(); ++kpt_idx)
        {
            cv::circle(octave_img,
                       cv::Point2d(octave_kpts(1, kpt_idx),
                                   octave_kpts(2, kpt_idx)),
                       3,
                       cv::Scalar(0, 0, 255));
        }

        std::stringstream s;
        s << "octave " << octave;
        show(octave_img, s.str());
    }
}

Eigen::VectorXd
weightedhistc(const Eigen::MatrixXd &vals,
              const Eigen::MatrixXd &weights,
              const Eigen::VectorXd &edges)
{
    Eigen::VectorXd hist = Eigen::VectorXd::Zero(edges.size() - 1);

    for (size_t val_idx = 0; val_idx < vals.size(); ++val_idx)
    {
        for (size_t hist_idx = 0; hist_idx < hist.size(); ++hist_idx)
        {
            if (vals(val_idx) >= edges(hist_idx) && vals(val_idx) <= edges(hist_idx + 1))
            {
                hist(hist_idx) += weights(val_idx);
                continue;
            }
        }
    }
    return hist;
}

Eigen::MatrixXd concat_h(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
    assert(a.rows() == b.rows());
    Eigen::MatrixXd res(a.rows(), a.cols() + b.cols());
    res << a, b;
    return res;
}

Eigen::MatrixXd concat_v(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
    assert(a.cols() == b.cols());
    Eigen::MatrixXd res(a.rows() + b.rows(), a.cols());
    res << a, b;
    return res;
}

std::vector<Eigen::VectorXd>
compute_descriptors(const std::vector<std::vector<Eigen::MatrixXd>> &blurred_images,
                    const std::vector<MatrixXS> &kpts,
                    bool rot_invariant,
                    size_t num_scales_in_octave,
                    std::vector<MatrixXS> &final_locations)
{
    assert(kpts.size() == blurred_images.size()); // number of octaves
    ImageGradient grads(blurred_images);
    auto gausswindow = gaussian_kernel(16 * 1.5, 16, 16);
    size_t border = 8;
    Eigen::VectorXd bin_edges = Eigen::VectorXd::LinSpaced(9, -1.0 * M_PI, M_PI);
    std::vector<Eigen::VectorXd> descs;
    for (size_t octave = 0; octave < kpts.size(); ++octave)
    {
        for (auto &kpt : kpts[octave].colwise())
        {
            size_t scale = kpt(0);
            size_t col = kpt(1);
            size_t row = kpt(2);

            size_t s_index = (scale - 1) - num_scales_in_octave * octave;
            auto &img = blurred_images[octave][s_index];
            if (col > img.cols() - border ||
                col < border ||
                row < border ||
                row > img.rows() - border)
            {
                continue;
            }

            auto &g_mag = grads.get_grad_mag(octave, s_index);
            auto &g_dir = grads.get_grad_dir(octave, s_index);

            size_t s_col = col - 7;
            size_t s_row = row - 7;
            Eigen::MatrixXd patch_mag = g_mag.block(s_row, s_col, 16, 16);
            Eigen::MatrixXd patch_dir = g_dir.block(s_row, s_col, 16, 16);
            Eigen::MatrixXd patch_mag_smoothed = (patch_mag.array() * gausswindow.array()).matrix();
            Eigen::VectorXd desc(128);
            for (size_t idx_col = 0; idx_col < 4; ++idx_col)
            {
                for (size_t idx_row = 0; idx_row < 4; ++idx_row)
                {
                    auto hist = weightedhistc(patch_dir.block(4 * idx_row, 4 * idx_col, 4, 4),
                                              patch_mag_smoothed.block(4 * idx_row, 4 * idx_col, 4, 4),
                                              bin_edges);
                    size_t index_in_desc = ((4 * idx_col) + idx_row) * 8;
                    desc.block(index_in_desc, 0, 8, 1) = hist;
                }
            }
            desc.normalize();
            descs.push_back(desc);

            MatrixXS loc(2, 1);
            loc << kpt(2) * std::pow(2, octave),
                kpt(1) * std::pow(2, octave); // row, col
            final_locations.push_back(loc);
        }
    }
    return descs;
}

std::map<size_t, std::tuple<size_t, double>>
match_features(const std::vector<std::vector<Eigen::VectorXd>> &descriptors,
               double max_ratio)
{
    assert(descriptors.size() == 2);
    assert(max_ratio <= 1.0 && max_ratio > 0);
    auto &descs1 = descriptors[0];
    auto &descs2 = descriptors[1];

    std::map<size_t, std::tuple<size_t, double>> matches;

    for (size_t idx1 = 0; idx1 < descs1.size(); ++idx1)
    {
        bool valid[2] = {false, false};
        size_t best_idx[2];
        double best_dist[2];

        auto &desc1 = descs1[idx1];
        for (size_t idx2 = 0; idx2 < descs2.size(); ++idx2)
        {
            auto &desc2 = descs2[idx2];
            double dist = (desc1 - desc2).norm();
            if (!valid[0])
            {
                valid[0] = true;
                best_idx[0] = idx2;
                best_dist[0] = dist;
            }
            else if (valid[0])
            {
                if (dist < best_dist[0])
                {
                    best_idx[1] = best_idx[0];
                    best_dist[1] = best_dist[0];
                    valid[1] = true;
                    best_idx[0] = idx2;
                    best_dist[0] = dist;
                }
                else if (!valid[1])
                {
                    best_idx[1] = idx2;
                    best_dist[1] = dist;
                    valid[1] = true;
                }
            }
            else if (valid[1])
            {
                if (dist < best_dist[1])
                {
                    best_idx[1] = idx2;
                    best_dist[1] = dist;
                }
            }
        }

        if (valid[0] && valid[1])
        {
            if (best_dist[1] > 1e-6) // non zero denominator
            {
                if (best_dist[0] / best_dist[1] > max_ratio)
                {
                    // reject the match
                    continue;
                }
            }
        }

        // accept the match and check for uniqueness
        // matches is the map from the index of feature in descs2 to
        // pair of < index in descs1, best_dist>
        auto iter = matches.find(best_idx[0]);
        if (iter != matches.end())
        {
            // There is already a match for the best_idx;
            double prev_dist;
            std::tie(std::ignore, prev_dist) = iter->second;
            if (best_dist[0] < prev_dist)
            {
                iter->second = std::make_tuple(idx1, best_dist[0]);
            }
        }
        else
        {
            // this match is not seen before;
            matches[best_idx[0]] = std::make_tuple(idx1, best_dist[0]);
        }

    }
    return matches;
}