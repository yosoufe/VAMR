#include "utils.hpp"
#include "folder_manager.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "keypoint_tracking.hpp"
#include "cuda_types.hpp"
#include "utils.hpp"

int main_cpu()
{
    std::string in_data_root = "../../data/ex03/";
    std::string out_data_root = "../../output/ex03/";
    SortedImageFiles image_files(in_data_root);

    cv::Size img_size;

    size_t patch_size = 9;
    double harris_kappa = 0.08;
    size_t non_maximum_suppression_radius = 9;
    size_t num_keypoints = 200;
    size_t descriptor_radius = 9;
    double match_lambda = 4;

    for (int i = 0; i < 10; ++i)
    {
        auto start_time = second();
        // Part 1: calculate corner response functions
        auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
        img_size = src_img.size();
        Eigen::MatrixXd eigen_img = cv_2_eigen(src_img);
        auto shi_tomasi_score = shi_tomasi(eigen_img, patch_size);
        auto harris_score = harris(eigen_img, patch_size, harris_kappa);
        // viz_harris_shi_tomasi_scores(src_img,
        //                              shi_tomasi_score, harris_score);

        // Part 2: Select keypoints
        auto shi_tomasi_kps = select_keypoints(shi_tomasi_score, num_keypoints, non_maximum_suppression_radius);
        auto harris_kps = select_keypoints(harris_score, num_keypoints, non_maximum_suppression_radius);
        // viz_key_points(src_img,
        //                shi_tomasi_score, harris_score,
        //                shi_tomasi_kps, harris_kps);

        // Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
        auto shi_tomasi_descriptors = describe_keypoints(eigen_img, shi_tomasi_kps, descriptor_radius);
        auto harris_descriptors = describe_keypoints(eigen_img, harris_kps, descriptor_radius);
        // viz_descriptors(src_img,
        //                 shi_tomasi_score, harris_score,
        //                 shi_tomasi_kps, harris_kps,
        //                 shi_tomasi_descriptors, harris_descriptors);
        std::cout << "elapsed CPU: " << second() - start_time << std::endl;
    }

    cv::VideoWriter video = create_video_writer(img_size, out_data_root + "keypoint_tracking.mp4");

    // Part 4 and 5 - Match descriptors between all images
    Eigen::MatrixXd prev_desc;
    Eigen::MatrixXd prev_kps;
    for (auto &image_path : image_files)
    {
        cv::Mat src_img = cv::imread(image_path.path(), cv::IMREAD_GRAYSCALE);
        Eigen::MatrixXd eigen_img = cv_2_eigen(src_img);
        auto harris_score = harris(eigen_img, patch_size, harris_kappa);
        auto curr_kps = select_keypoints(harris_score, num_keypoints, non_maximum_suppression_radius);
        auto desc = describe_keypoints(eigen_img, curr_kps, descriptor_radius);

        if (prev_desc.size() != 0)
        {
            auto matches = match_descriptors(desc, prev_desc, match_lambda);
            video << viz_matches(src_img, matches, curr_kps, prev_kps);
        }
        prev_desc = desc;
        prev_kps = curr_kps;
    }

    return 0;
}

int main_gpu()
{
    std::string in_data_root = "../../data/ex03/";
    std::string out_data_root = "../../output/ex03/";
    SortedImageFiles image_files(in_data_root);

    cv::Size img_size;

    size_t patch_size = 9;
    double harris_kappa = 0.08;
    size_t non_maximum_suppression_radius = 9;
    size_t num_keypoints = 200;
    size_t descriptor_radius = 9;
    double match_lambda = 4;

    for (int i = 0; i < 10; ++i)
    {
        auto start_time = second();
        // Part 1: calculate corner response functions
        auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
        img_size = src_img.size();
        Eigen::MatrixXd eigen_img = cv_2_eigen(src_img);
        auto cuda_eigen_img = cuda::eigen_to_cuda(eigen_img);
        auto shi_tomasi_score = cuda::shi_tomasi(cuda_eigen_img, patch_size);
        auto harris_score = cuda::harris(cuda_eigen_img, patch_size, harris_kappa);

        // Part 2: Select keypoints
        auto shi_tomasi_kps = cuda::select_keypoints(shi_tomasi_score, non_maximum_suppression_radius);
        auto harris_kps = cuda::select_keypoints(harris_score, non_maximum_suppression_radius);

        // Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
        auto shi_tomasi_descriptors = cuda::describe_keypoints(
            cuda_eigen_img,
            shi_tomasi_kps,
            num_keypoints,
            descriptor_radius);
        auto harris_descriptors = cuda::describe_keypoints(
            cuda_eigen_img,
            harris_kps,
            num_keypoints,
            descriptor_radius);

        // copy to cpu for visualization
        // Yet, I don't know how to visualize directly from cuda.
        auto h_shi_tomasi_score = cuda::cuda_to_eigen(shi_tomasi_score);
        auto h_harris_score = cuda::cuda_to_eigen(harris_score);
        auto h_shi_tomasi_kps = cuda::cuda_to_eigen(shi_tomasi_kps).block(0, 0, 2, num_keypoints);
        auto h_harris_kps = cuda::cuda_to_eigen(harris_kps).block(0, 0, 2, num_keypoints);
        auto h_shi_tomasi_descriptor = cuda::cuda_to_eigen(shi_tomasi_descriptors);
        auto h_harris_descriptors = cuda::cuda_to_eigen(harris_descriptors);
        // viz_descriptors(src_img,
        //                 h_shi_tomasi_score, h_harris_score,
        //                 h_shi_tomasi_kps, h_harris_kps,
        //                 h_shi_tomasi_descriptor, h_shi_tomasi_descriptions);
        
        std::cout << "elapsed gpu: " << second() - start_time << std::endl;
    }

    return 0;
}

int main()
{
    // return main_cpu();
    return main_gpu();
}