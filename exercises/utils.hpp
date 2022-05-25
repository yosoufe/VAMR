#pragma once

#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <string>
#include <cassert>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry> // AngleAxis
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/eigen.hpp>

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

cv::Mat
load_image_color(std::string image_path);

void
draw_circles(cv::Mat &src_img, const Eigen::Matrix2Xd &pts, int thinkness, const cv::Scalar &color = cv::Scalar(0, 0, 255), int lineType=cv::FILLED);

void draw_cube(cv::Mat &src_img, const Eigen::Matrix2Xd &pts);


cv::VideoWriter
create_video_writer(const cv::Size &img_size, const std::string &file_path);

cv::Mat convet_to_cv_to_show(const Eigen::MatrixXd& eigen_img);

Eigen::MatrixXd cv_2_eigen(const cv::Mat &img);

cv::Mat eigen_2_cv(const Eigen::MatrixXd &eigen);

Eigen::MatrixXd correlation(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel);

void show(const cv::Mat &img, std::string window_name = "image");
