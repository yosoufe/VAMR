#include "operations.hpp"
#include "counting_iterator.hpp"

Eigen::MatrixXd correlation(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel)
{
    size_t kernel_rv = kernel.rows() / 2;
    size_t kernel_sv = kernel.rows();

    size_t kernel_ru = kernel.cols() / 2;
    size_t kernel_su = kernel.cols();

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(input.rows(), input.cols());

    std::for_each_n(
        std::execution::par_unseq, 
        counting_iterator(kernel_rv),
        input.rows() - kernel_rv - kernel_rv,
        [&](size_t v)
        {
            std::for_each_n(
                std::execution::par_unseq,
                counting_iterator(kernel_ru),
                input.cols() - kernel_ru - kernel_ru,
                [&](size_t u)
                {
                    auto element_wise_prod = input.block(v - kernel_rv, u - kernel_ru, kernel_sv, kernel_su).array() * kernel.array();
                    res(v, u) = element_wise_prod.sum();
                }
            );
        }
    );
    return res;
}

Eigen::MatrixXd sobel_x_kernel()
{
    return Eigen::Matrix3d(
        {{-1.0, 0.0, 1.0},
         {-2.0, 0.0, 2.0},
         {-1.0, 0.0, 1.0}});
}

Eigen::MatrixXd sobel_y_kernel()
{
    return Eigen::Matrix3d(
        {{-1.0, -2.0, -1.0},
         {0.0, 0.0, 0.0},
         {1.0, 2.0, 1.0}});
}