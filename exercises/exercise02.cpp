#include "utils.hpp"
#include <opencv2/core/eigen.hpp>

#include "pnp_dlt.hpp"

int main(int argc, char **argv)
{
    std::string in_data_root = "../../data/ex02/";
    std::string out_data_root = "../../output/ex02/";

    auto K = read_K_matrix(in_data_root + "K.txt");
    std::cout << "K matrix: \n"
              << K << "\n"
              << std::endl;

    Eigen::MatrixXd points_in_W = read_matrix(in_data_root + "p_W_corners.txt", ',').transpose();
    std::cout << "Points in world coordinate: \n"
              << points_in_W << "\n"
              << std::endl;

    auto all_detected_corners = read_detected_corners(in_data_root + "detected_corners.txt");

    // load single image to get image size
    auto image = load_image_color(in_data_root + "images_undistorted/img_0001.jpg");
    auto img_size = image.size();
    cv::VideoWriter re_projected_vid_writer = create_video_writer(
        img_size,
        out_data_root+"reprojected_points.mp4");

    std::vector<Eigen::MatrixXd> poses;
    std::vector<cv::Mat> images;

    for (size_t image_idx = 0; image_idx < 210; image_idx++)
    {
        auto &detected_corners = all_detected_corners[image_idx];

        std::stringstream image_path;
        image_path << "images_undistorted/img_" << std::setfill('0') << std::setw(4) << image_idx + 1 << ".jpg";
        image = load_image_color(in_data_root + image_path.str());

        draw_circles(image, detected_corners.block(0, 0, 2, detected_corners.cols()), 3, cv::Scalar(0, 0, 255), cv::LINE_8);

        Eigen::MatrixXd pose = estimate_pose_dlt(detected_corners, points_in_W, K);
        Eigen::MatrixXd re_projected_points = re_project_points(points_in_W, pose, K);
        // std::cout << "differences between original and re-projected:" << std::endl
        //           << re_projected_points - detected_corners.block(0, 0, 2, 12) << std::endl;
        poses.push_back(pose);

        draw_circles(image, re_projected_points, 2, cv::Scalar(255, 0, 0), cv::LINE_4);

        re_projected_vid_writer << image;
        images.push_back(image);
    }

    plot_trajectory_3d(poses, images, points_in_W, out_data_root+"output_vid.mp4");

    return 0;
}