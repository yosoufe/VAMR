#include "sift.hpp"
#include <gtest/gtest.h>

TEST(SiftTest, sift_sigmas)
{
    std::vector<size_t> num_scales = {1, 2, 3, 4, 5};
    for (auto n_scales : num_scales)
    {
        std::vector<double> sigmas = sift_sigmas(n_scales, 1.6);
        EXPECT_EQ(sigmas.size(), n_scales + 3);
        // for (auto sigma : sigmas)
        //     std::cout << sigma << " ";
        // std::cout << std::endl;
    }
}

TEST(SiftTest, gaussian_vector)
{
    double sigma = std::sqrt(1.0/(2*std::log(2.0)));
    auto v = gaussian_vector(sigma, 1);
    std::cout << "gaussian filter for sigma: " << sigma << " is ";
    std::cout << v.transpose() << std::endl;
}


TEST(SiftTest, gaussian_generation)
{
    std::vector<double> sigmas = sift_sigmas(5, 1.6);
    sigmas.push_back(1.0);
    std::vector<int> sigma_vector_sizes = {11, 11, 13, 15, 17, 19, 21, 25, 7};

    for (size_t idx; idx < sigmas.size(); idx++)
    {
        size_t expected_size = sigma_vector_sizes[idx];
        double sigma = sigmas[idx];

        auto kernel_vector = gaussian_vector(sigma);
        EXPECT_EQ(expected_size, kernel_vector.rows());

        // std::cout << "-------" << sigma << "---------" << std::endl;
        // std::cout << "[" << kernel_vector.transpose()
        //           << "] sum = " << kernel_vector.sum()
        //           << std::endl;
    }
}

TEST(SiftTest, CalculateDoGs)
{
    Eigen::MatrixXd eigen_octave_img = Eigen::MatrixXd::Random(100, 100);
    std::vector<Eigen::MatrixXd> DoGs;
    calculate_DoGs(
        1, eigen_octave_img, DoGs, 1.6);
    EXPECT_NE(DoGs.size(), 0);
    EXPECT_EQ(DoGs.size(), 3);
    for (auto &DoG : DoGs)
    {
        EXPECT_EQ(DoG.rows(), 100);
        EXPECT_EQ(DoG.cols(), 100);
    }
}

TEST(SiftTest, find_keypoints)
{
    std::vector<Eigen::MatrixXd> DoGs;
    DoGs.push_back(Eigen::MatrixXd::Zero(20, 20));
    DoGs.push_back(Eigen::MatrixXd::Ones(20, 20));
    DoGs.push_back(Eigen::MatrixXd::Zero(20, 20));
    DoGs[1](11, 10) = 5;
    MatrixXS res = find_keypoints(DoGs, 4, 1);
    EXPECT_EQ(res.cols(), 1);
    EXPECT_EQ(res.rows(), 4);
    EXPECT_EQ(res(0, 0), 1);  // scale
    EXPECT_EQ(res(1, 0), 0);  // ocatve
    EXPECT_EQ(res(2, 0), 10); // u
    EXPECT_EQ(res(3, 0), 11); // v
}