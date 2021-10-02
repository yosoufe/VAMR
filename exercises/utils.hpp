#pragma once

#include <iostream>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/Geometry> // AngleAxis
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

std::ifstream read_file(std::string path);

template <class T>
void print_shape(T m)
{
    std::cout << m.rows() << " x " << m.cols() << std::endl;
};

Eigen::MatrixXd
read_matrix(std::string file_path, char delimiter=' ');

std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
read_pose_file(std::string file_path);

void
read_distortion_param(std::string file_path, double &k1, double &k2);

Eigen::Matrix3d
read_K_matrix(std::string file_path);

Eigen::MatrixXd
create_grid(double cell_size, size_t num_x, size_t num_y);

Eigen::MatrixXd
create_cube(double cell_size = 0.4);

Eigen::Matrix2Xd
distorted_pixel(const double k1,
                const double k2,
                const Eigen::Vector2d &principal_pt,
                const Eigen::Matrix2Xd &points);

Eigen::Matrix2Xd
project_2_camera_frame(const Eigen::Matrix3d &intrinsics,
                       const Eigen::Isometry3d &extrinsics,
                       const Eigen::Matrix3Xd &points);

cv::Mat
load_image(std::string image_path);

void
draw_circles(cv::Mat &src_img, const Eigen::Matrix2Xd &pts, int thinkness, const cv::Scalar &color = cv::Scalar(0, 0, 255), int lineType=cv::FILLED);

void draw_cube(cv::Mat &src_img, const Eigen::Matrix2Xd &pts);

cv::Mat
undistort_image(const cv::Mat &src_img,
                double d1,
                double d2,
                const Eigen::Vector2d &principal_pt);

cv::VideoWriter
create_video_writer(const cv::Size &img_size, const std::string &file_path);