#include <iostream>
#include <fstream> // ifstream
#include <Eigen/Core>
#include <Eigen/Geometry> // AngleAxis
#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

std::ifstream read_file(std::string path)
{
    std::ifstream fin(path);
    if (!fin || !fin.is_open())
    {
        std::stringstream error_msg;
        error_msg << "cannot find pose file at " << path << std::endl;
        std::cout << error_msg.str();
    }
    return fin;
}

template <class T>
void print_shape(T m)
{
    std::cout << m.rows() << " x " << m.cols() << std::endl;
}

auto read_pose_file(std::string file_path)
{
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    auto fin = read_file(file_path);
    if (!fin)
    {
        return poses;
    }

    // read pose file
    while (!fin.eof())
    {
        double wx, wy, wz, tx, ty, tz;
        fin >> wx >> wy >> wz >> tx >> ty >> tz;
        Eigen::Vector3d rot(wx, wy, wz);
        double angle = rot.norm();
        Eigen::AngleAxisd rot_angleaxis(angle, rot / angle);

        Eigen::Isometry3d Twr(rot_angleaxis);
        Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
        poses.push_back(Twr);
    }
    return poses;
}

void read_distortion_param(std::string file_path, double &k1, double &k2)
{
    auto fin = read_file(file_path);
    if (!fin)
    {
        return;
    }
    fin >> k1 >> k2;
    return;
}

auto read_K_matrix(std::string file_path)
{
    auto fin = read_file(file_path);
    if (!fin)
    {
        return Eigen::Matrix3d();
    }
    std::vector<double> buf;
    size_t counter = 0;
    while (!fin.eof() && counter < 9)
    {
        double v = 0;
        fin >> v;
        buf.push_back(v);
        counter++;
    }
    Eigen::Matrix3d k_matrix(buf.data());
    k_matrix.resize(3, 3);
    return Eigen::Matrix3d(k_matrix.transpose());
}

Eigen::MatrixXd create_grid(double cell_size, size_t num_x, size_t num_y)
{
    size_t num_grid_points = num_x * num_y;
    Eigen::MatrixXd grid;
    grid.resize(3, num_grid_points);
    for (size_t idx = 0; idx < num_grid_points; idx++)
    {
        double x = idx % num_x;
        double y = idx / num_x;
        grid.block(0, idx, 3, 1) << x * cell_size, y * cell_size, 0.0d;
    }
    return grid;
}

auto distorted_pixel(const double k1,
                     const double k2,
                     const Eigen::Vector2d &principal_pt,
                     const Eigen::Matrix2Xd &points)
{
    // TODO
    auto d_pts = points.colwise() - principal_pt;
    auto r_square = d_pts.colwise().squaredNorm();
    auto term_c = ((k1 * r_square).array() + k2 * (r_square.array().pow(2)) + 1.0);
    return (d_pts.array().rowwise() * term_c).matrix().colwise() + principal_pt;
}

Eigen::Matrix2Xd project_2_camera_frame(const Eigen::Matrix3d &intrinsics,
                                        const Eigen::Isometry3d &extrinsics,
                                        const Eigen::Matrix3Xd &points)
{
    Eigen::Matrix2Xd res;
    res.resize(2, points.cols());
    for (int col_idx = 0; col_idx < points.cols(); col_idx++)
    {
        auto homoG = intrinsics * extrinsics * Eigen::Vector3d(points.block(0, col_idx, 3, 1));
        res.block(0, col_idx, 2, 1) = homoG.block(0, 0, 2, 1) / homoG(2);
    }
    return res;
}

cv::Mat load_image(std::string image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    // std::cout << img.dims << std::endl;
    return img;
}

void draw_circles(cv::Mat &src_img, const Eigen::Matrix2Xd &pts, int thinkness)
{
    int num_pts = pts.cols();
    for (size_t pt_idx = 0; pt_idx < num_pts; pt_idx++)
    {
        cv::circle(src_img, cv::Point2d(pts(0,pt_idx),pts(1,pt_idx)), thinkness, cv::Scalar( 0, 0, 255 ), cv::FILLED);
    }
}

int main(int argc, char **argv)
{
    std::string in_data_root = "../../data/ex01/";
    std::ctring out_data_root = "../../output/ex01";
    auto poses = read_pose_file(in_data_root + "poses.txt");
    if (poses.size() == 0)
    {
        return -1;
    }
    auto grid = create_grid(0.04, 9, 6);
    // std::cout << grid << std::endl;
    auto K = read_K_matrix(in_data_root + "K.txt");
    // std::cout << K << std::endl;
    double d1 = 0, d2 = 0;
    read_distortion_param(in_data_root + "D.txt", d1, d2);
    // std::cout << d1 << " " << d2 << std::endl;

    // print_shape(grid);

    Eigen::Vector2d principal_pt = K.block(0, 2, 2, 1);

    auto pts_in_camera_frame = project_2_camera_frame(K, poses[0], grid);
    // std::cout << pts_in_camera_frame.block(0,0,2,5) << std::endl;

    auto pts_in_distorted_img = distorted_pixel(d1, d2, principal_pt, pts_in_camera_frame);

    // load image
    auto image = load_image(in_data_root + "images/img_0001.jpg");

    // draw circles on points on the image
    draw_circles(image, pts_in_distorted_img, 3);

    // save the image
    cv::

    // create a movie of all images.

    // show the image
    cv::imshow("Display window", image);
    cv::waitKey(0); // Wait for a keystroke in the window
}