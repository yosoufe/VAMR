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

TEST(SiftTest, gaussian_kernel)
{
    double sigma = std::sqrt(1.0/(2*std::log(2.0)));
    auto v = gaussian_kernel(sigma, 1);
    std::cout << "gaussian filter for sigma: " << sigma << " is ";
    std::cout << v.transpose() << std::endl;
}


TEST(SiftTest, gaussian_generation)
{
    std::vector<double> sigmas = sift_sigmas(5, 1.6);
    sigmas.push_back(1.0);
    std::vector<int> sigma_vector_sizes = {7, 7, 9, 9, 11, 13, 13, 15, 5};

    for (size_t idx; idx < sigmas.size(); idx++)
    {
        size_t expected_size = sigma_vector_sizes[idx];
        double sigma = sigmas[idx];

        auto kernel_vector = gaussian_kernel(sigma);
        EXPECT_EQ(expected_size, kernel_vector.rows());

        // std::cout << "-------" << sigma << "---------" << std::endl;
        // std::cout << "[" << kernel_vector.transpose()
        //           << "] sum = " << kernel_vector.sum()
        //           << std::endl;
    }
}

// TODO: fix this
// TEST(SiftTest, CalculateDoGs)
// {
//     Eigen::MatrixXd eigen_octave_img = Eigen::MatrixXd::Random(100, 100);
//     std::vector<Eigen::MatrixXd> DoGs;
//     calculate_DoGs(
//         1, eigen_octave_img, DoGs, 1.6);
//     EXPECT_NE(DoGs.size(), 0);
//     EXPECT_EQ(DoGs.size(), 3);
//     for (auto &DoG : DoGs)
//     {
//         EXPECT_EQ(DoG.rows(), 100);
//         EXPECT_EQ(DoG.cols(), 100);
//     }
// }

TEST(SiftTest, find_keypoints)
{
    std::vector<Eigen::MatrixXd> DoGs;
    DoGs.push_back(Eigen::MatrixXd::Zero(20, 20));
    DoGs.push_back(Eigen::MatrixXd::Ones(20, 20));
    DoGs.push_back(Eigen::MatrixXd::Zero(20, 20));
    DoGs[1](11, 10) = 5;
    std::vector<std::vector<Eigen::MatrixXd>> DoGs_all;
    DoGs_all.push_back(DoGs);
    auto kpts = extract_keypoints(DoGs_all, 4);
    MatrixXS res = kpts[0];
    EXPECT_EQ(res.cols(), 1);
    EXPECT_EQ(res.rows(), 3);
    EXPECT_EQ(res(0, 0), 1);  // scale
    EXPECT_EQ(res(1, 0), 10);  // ocatve
    EXPECT_EQ(res(2, 0), 11); // u
}


TEST(SiftTest, weightedhistc)
{
    Eigen::MatrixXd vals(2,2);
    vals << 1,2,3,4;
    Eigen::MatrixXd weights(2,2);
    weights << 1,1,1,1;
    Eigen::VectorXd bin_edges = Eigen::VectorXd::LinSpaced(3, 1, 4);
    std::cout << bin_edges.transpose() << std::endl;
    auto hist = weightedhistc(vals, weights, bin_edges);
    std::cout << "hist:\n" << hist.transpose() << std::endl;
}
