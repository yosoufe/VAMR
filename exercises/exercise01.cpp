#include "utils.hpp"

int main(int argc, char **argv)
{
    std::string in_data_root = "../../data/ex01/";
    auto poses = read_pose_file(in_data_root + "poses.txt");
    if (poses.size() == 0)
    {
        return -1;
    }
    auto grid = create_grid(0.04, 9, 6);
    auto cube = create_cube(0.04 * 2);
    // std::cout << grid << std::endl;
    auto K = read_K_matrix(in_data_root + "K.txt");
    // std::cout << K << std::endl;
    double d1 = 0, d2 = 0;
    read_distortion_param(in_data_root + "D.txt", d1, d2);
    // std::cout << d1 << " " << d2 << std::endl;

    // print_shape(grid);
    Eigen::Vector2d principal_pt = K.block(0, 2, 2, 1);

    // load single image to get image size
    auto image = load_image_color(in_data_root + "images/img_0001.jpg");
    auto img_size = image.size();

    cv::VideoWriter grid_video_distorted = create_video_writer(img_size, "ex01/distorted_grid.mp4");
    cv::VideoWriter video_undistorted = create_video_writer(img_size, "ex01/undistorted.mp4");

    for (size_t image_idx = 1; image_idx <= 736; image_idx++)
    {
        std::stringstream image_path;
        image_path << "images/img_" << std::setfill('0') << std::setw(4) << image_idx << ".jpg";
        image = load_image_color(in_data_root + image_path.str());

        // make a video of grid points on distorted images
        auto grid_in_camera_frame = project_2_camera_frame(K, poses[image_idx - 1], grid);
        auto grid_in_distorted_img = distorted_pixel(d1, d2, principal_pt, grid_in_camera_frame);
        // draw circles on points on the image
        draw_circles(image, grid_in_distorted_img, 3);
        grid_video_distorted << image;

        // make a video of grid points on distorted images
        cv::Mat undistorted_img = undistort_image(image, d1, d2, principal_pt);
        auto cube_in_camera_frame = project_2_camera_frame(K, poses[image_idx - 1], cube);
        draw_cube(undistorted_img, cube_in_camera_frame);
        video_undistorted << undistorted_img;
    }

    std::cout << "Finished writing" << std::endl;
    return 0;
}