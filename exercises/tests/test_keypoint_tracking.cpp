#include "keypoint_tracking.hpp"
#include <gtest/gtest.h>

TEST(keypoint_tracking, calculate_Is)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(3, 3);
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;

    input << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    calculate_Is(input, 3, sI_xx, sI_yy, sI_xy);

    Eigen::MatrixXd expected_I = Eigen::MatrixXd::Zero(3, 3);
    expected_I(1, 1) = 8 * 8;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_xx));

    expected_I(1, 1) = 24 * 24;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_yy));

    expected_I(1, 1) = 24 * 8;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_xy));
}

TEST(keypoint_tracking, non_maximum_suppression_cpu)
{
    Eigen::MatrixXd input(4, 5);
    input << 10, 1, 3, 6, 10,
        2, 3, 7, 3, 1,
        5, 4, 4, 1, 2,
        4, 1, 9, 0, 3;
    int patch_size = 3;

    Eigen::MatrixXd expected(4, 5);
    expected << 10, 0, 0, 0, 10,
        0, 0, 7, 0, 0,
        5, 0, 0, 0, 0,
        0, 0, 9, 0, 3;

    auto output = non_maximum_suppression(input, patch_size);
    EXPECT_TRUE(are_matrices_close(output, expected));
    // std::cout << output << std::endl;
}

#if WITH_CUDA

TEST(keypoint_tracking, calculate_Is_gpu)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(15, 16);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        Eigen::MatrixXd h_sI_xx, h_sI_yy, h_sI_xy;
        cuda::CuMatrixD d_sI_xx, d_sI_yy, d_sI_xy;
        size_t patch_size = 9;
        calculate_Is(h_img, patch_size, h_sI_xx, h_sI_yy, h_sI_xy);
        cuda::calculate_Is(d_img, patch_size, d_sI_xx, d_sI_yy, d_sI_xy);
        auto hd_sI_xx = cuda::cuda_to_eigen(d_sI_xx);
        auto hd_sI_yy = cuda::cuda_to_eigen(d_sI_yy);
        auto hd_sI_xy = cuda::cuda_to_eigen(d_sI_xy);

        EXPECT_TRUE(are_matrices_close(hd_sI_xx,
                                       h_sI_xx));
        EXPECT_TRUE(are_matrices_close(hd_sI_yy,
                                       h_sI_yy));
        EXPECT_TRUE(are_matrices_close(hd_sI_xy,
                                       h_sI_xy));
        // std::cout << "sI_xx CPU\n"
        //           << h_sI_xx << std::endl<< std::endl;
        // std::cout << "sI_xx GPU\n"
        //           << hd_sI_xx << std::endl;
    }
}

TEST(keypoint_tracking, harris_score)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        size_t patch_size = 5;
        double kappa = 0.08;
        auto h_score = harris(h_img, patch_size, kappa);
        auto d_score = harris(d_img, patch_size, kappa);
        auto hd_score = cuda::cuda_to_eigen(d_score);
        EXPECT_TRUE(are_matrices_close(hd_score,
                                       h_score));

        // std::cout << "h_score CPU\n"
        //           << h_score << std::endl;
        // std::cout << "d_score GPU\n"
        //           << hd_score << std::endl;
    }
}

TEST(keypoint_tracking, shi_tomasi_score)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        size_t patch_size = 5;
        auto h_score = shi_tomasi(h_img, patch_size);
        auto d_score = shi_tomasi(d_img, patch_size);
        auto hd_score = cuda::cuda_to_eigen(d_score);
        EXPECT_TRUE(are_matrices_close(hd_score,
                                       h_score));

        // std::cout << "h_score CPU\n"
        //           << h_score << std::endl;
        // std::cout << "d_score GPU\n"
        //           << hd_score << std::endl;
    }
}

void debug_non_maximum_suppression(cuda::CuMatrixD d_output, Eigen::MatrixXd expected)
{

    auto hd_output = cuda::cuda_to_eigen(d_output);
    Eigen::MatrixXd diff = hd_output - expected;
    auto non_zero_indicies = find_non_zero_indicies(diff);
    std::cout << "printing " << non_zero_indicies.size() << " indicies\n";
    for (auto &idx : non_zero_indicies)
        printf("(%ld, %ld, %f = %f - %f)\n",
               idx % expected.rows(),
               idx / expected.rows(),
               diff(idx),
               hd_output(idx),
               expected(idx));

    std::cout << "\ndiff\n"
              << diff << std::endl;
    std::cout << "\nexpected\n"
              << expected << std::endl;
    std::cout << "\noutput\n"
              << hd_output << std::endl;
}

void test_non_maximum_suppression(Eigen::MatrixXd input, int patch_size, bool debug = false)
{
    auto start = second();
    Eigen::MatrixXd expected = non_maximum_suppression(input, patch_size);
    printf("\n\nCPU time %f\n", second() - start);

    auto d_input = cuda::eigen_to_cuda(input);

    start = second();
    auto output = cuda::non_maximum_suppression_1(d_input, patch_size);
    printf("GPU time using global memory %f\n", second() - start);
    EXPECT_TRUE(are_matrices_close(output, expected));

    start = second();
    output = cuda::non_maximum_suppression_2(d_input, patch_size);
    printf("GPU time using shared memory1 %f\n", second() - start);
    EXPECT_TRUE(are_matrices_close(output, expected));

    start = second();
    output = cuda::non_maximum_suppression_3(d_input, patch_size);
    printf("GPU time using shared memory2 %f\n", second() - start);
    EXPECT_TRUE(are_matrices_close(output, expected));

    if (debug)
        debug_non_maximum_suppression(output, expected);
}

TEST(keypoint_tracking, non_maximum_suppression)
{
    Eigen::MatrixXd input(4, 5);
    input << 10, 1, 3, 6, 10,
        2, 3, 7, 3, 1,
        5, 4, 4, 1, 2,
        4, 1, 9, 0, 3;
    test_non_maximum_suppression(input, 3);
    test_non_maximum_suppression(Eigen::MatrixXd::Random(3000, 4000).array() + 1, 3, false);
    for (int i = 0; i < 10; ++i)
        test_non_maximum_suppression(Eigen::MatrixXd::Random(1025, 800).array() + 1, 3, false);
}

void test_sort(Eigen::MatrixXd &input)
{
    std::cout << "\ninput before sort\n"
              << input << std::endl;
    auto d_input = cuda::eigen_to_cuda(input);
    cuda::CuMatrixD indicies;
    auto d_output = cuda::sort_matrix(d_input, indicies);
    // std::cout << "before sort\n" << cuda::cuda_to_eigen(d_input) << std::endl;
    std::cout << "after sort\n"
              << cuda::cuda_to_eigen(d_output) << std::endl;
    std::cout << "indicies\n"
              << cuda::cuda_to_eigen(indicies) << std::endl;
}

TEST(keypoint_tracking, sort_matrix)
{
    Eigen::MatrixXd input(4, 5);
    for (int i = 0; i < input.size(); ++i)
        input(i) = i;
    test_sort(input);
}

TEST(keypoint_tracking, select_keypoint_gpu)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 4).array() + 1;
    int num = 2;
    int radius = 1;
    auto expected = select_keypoints(input, num, radius);

    auto d_input = cuda::eigen_to_cuda(input);
    auto output = cuda::select_keypoints(d_input, radius);
    EXPECT_TRUE(are_matrices_close(output.block(0, 0, 2, num), expected));

    // std::cout << "input\n" << input << std::endl;
    // std::cout << "expected\n" << expected << std::endl;
    // std::cout << "output\n" << cuda::cuda_to_eigen(output) << std::endl;
}

TEST(keypoint_tracking, describe_keypoints)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1800, 900).array() + 1;
    int num = 6;
    int radius = 5;
    int descriptor_radius = 1;

    auto keypoints = select_keypoints(input, num, radius);
    auto descriptors = describe_keypoints(input, keypoints, descriptor_radius);

    auto d_input = cuda::eigen_to_cuda(input);
    auto d_keypoints = cuda::select_keypoints(d_input, radius);
    auto d_descriptors = cuda::describe_keypoints(d_input, d_keypoints, num, descriptor_radius);

    auto hd_descriptors = cuda::cuda_to_eigen(d_descriptors);
    EXPECT_TRUE(are_matrices_close(hd_descriptors, descriptors));

    // print_shape(descriptors);
    // print_shape(hd_descriptors);
    // std::cout << "keypoints\n"
    //           << keypoints << std::endl;
    // std::cout << "expected\n"
    //           << descriptors << std::endl;
    // std::cout << "output\n"
    //           << hd_descriptors << std::endl;
}

#endif