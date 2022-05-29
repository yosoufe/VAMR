#include "utils.hpp"
#include "camera_model.hpp"
#include "folder_manager.hpp"
#include <chrono>

int main(int argc, char **argv)
{
    std::string in_data_root = "../../data/ex01/";
    std::string out_data_root = "../../output/ex01/";
    SortedImageFiles image_files(in_data_root+"images");
    auto poses = read_pose_file(in_data_root + "poses.txt");
    
    if (poses.size() == 0) return -1;

    auto grid = create_grid(0.04, 9, 6);
    auto cube = create_cube(0.04 * 2);
    auto K = read_K_matrix(in_data_root + "K.txt");
    double d1 = 0, d2 = 0;
    read_distortion_param(in_data_root + "D.txt", d1, d2);
    Eigen::Vector2d principal_pt = K.block(0, 2, 2, 1);

    // load single image to get image size
    auto image = load_image_color(in_data_root + "images/img_0001.jpg");
    auto img_size = image.size();

    cv::VideoWriter grid_video_distorted = create_video_writer(img_size, out_data_root+"distorted_grid.mp4");
    cv::VideoWriter video_undistorted = create_video_writer(img_size, out_data_root+"undistorted.mp4");

    for (auto &image_path : image_files)
    {
        image = load_image_color(image_path.path());

        // make a video of grid points on distorted images
        auto grid_in_camera_frame = project_2_camera_frame(K, poses[image_path.number() - 1], grid);
        auto grid_in_distorted_img = distorted_pixel(d1, d2, principal_pt, grid_in_camera_frame);
        // draw circles on points on the image
        draw_circles(image, grid_in_distorted_img, 3);
        grid_video_distorted << image;

        // make a video of grid points on distorted images
        auto start = std::chrono::high_resolution_clock::now();
        #ifdef WITH_CUDA
            cv::Mat undistorted_img = cuda::undistort_image(image, d1, d2, principal_pt);
            // 2X speed up without any cuda code optimization.
        #else
            cv::Mat undistorted_img = undistort_image(image, d1, d2, principal_pt);
        #endif
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "duration in microseconds: " << duration.count() << std::endl;
        
        auto cube_in_camera_frame = project_2_camera_frame(K, poses[image_path.number() - 1], cube);
        draw_cube(undistorted_img, cube_in_camera_frame);
        video_undistorted << undistorted_img;
    }

    std::cout << "Finished writing" << std::endl;
    return 0;
}