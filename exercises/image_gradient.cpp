#include "image_gradient.hpp"
#include <math.h>
#include "operations.hpp"

ImageGradient::ImageGradient(const std::vector<std::vector<Eigen::MatrixXd>> &blurred_images) : blurred(blurred_images)
{
    sobel_kernel_y = sobel_y_kernel();
    sobel_kernel_x = sobel_kernel_y.transpose();
}

std::tuple<size_t, size_t> ImageGradient::make_key(size_t octave, size_t scale)
{
    return std::make_tuple(octave, scale);
}

double my_atan2(double a, double b)
{
    double res = std::atan2(b, a);
    // std::cout << "atan2(" << b << ", " << a << ") = " << res << std::endl;
    return res;
}

void ImageGradient::calculate_grads(size_t octave, size_t scale)
{
    auto key = make_key(octave, scale);
    Eigen::MatrixXd sobel_x = correlation(blurred[octave][scale], sobel_kernel_x);
    Eigen::MatrixXd sobel_y = correlation(blurred[octave][scale], sobel_kernel_y);

    g_mags[key] = (sobel_x.array().square() + sobel_y.array().square()).sqrt().matrix();
    g_dirs[key] = sobel_x.binaryExpr(sobel_y, std::ptr_fun(my_atan2)); // eigen does not have atan2.
}

Eigen::MatrixXd &ImageGradient::get_grad_mag(size_t octave, size_t scale)
{
    auto key = make_key(octave, scale);
    if (g_mags.find(key) == g_mags.end())
    {
        calculate_grads(octave, scale);
    }
    return g_mags[key];
}

Eigen::MatrixXd &ImageGradient::get_grad_dir(size_t octave, size_t scale)
{
    auto key = make_key(octave, scale);
    if (g_dirs.find(key) == g_dirs.end())
    {
        calculate_grads(octave, scale);
    }
    return g_dirs[key];
}