#include "utils.hpp"
#include <fstream> // ifstream

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

Eigen::MatrixXd read_matrix(std::string file_path, char delimiter)
{
    std::vector<std::vector<double>> data;

    auto fin = read_file(file_path);
    if (!fin)
    {
        throw std::runtime_error("problem in reading a file");
    }

    std::string line;
    while (std::getline(fin, line))
    {
        std::vector<double> row;
        std::stringstream line_stream(line);

        while (line_stream)
        {
            std::string section;
            if (!std::getline(line_stream, section, delimiter))
                break;
            if (section.empty()) continue; // ignore multiple delimiter
            row.push_back(std::stod(section.c_str()));
        }
        data.push_back(row);
    }
    int rows = data.size(), cols = data[0].size();

    Eigen::MatrixXd res;
    res.resize(rows, cols);
    for (size_t row_idx = 0; row_idx < rows; row_idx++)
    {
        res.row(row_idx) = Eigen::VectorXd::Map(&data[row_idx][0], cols);
    }
    return res;
}

std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
read_pose_file(std::string file_path)
{
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    auto mat = read_matrix(file_path);

    for (auto row : mat.rowwise())
    {
        Eigen::Vector3d rot(row[0], row[1], row[2]);
        double angle = rot.norm();
        Eigen::AngleAxisd rot_angleaxis(angle, rot / angle);

        Eigen::Isometry3d Twr(rot_angleaxis);
        Twr.pretranslate(Eigen::Vector3d(row[3], row[4], row[5]));
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

Eigen::Matrix3d
read_K_matrix(std::string file_path)
{
    auto mat = read_matrix(file_path, ' ');
    return Eigen::Matrix3d(mat);
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

Eigen::MatrixXd create_cube(double cell_size)
{
    size_t num_grid_points = 8;
    Eigen::MatrixXd cube;
    cube.resize(3, num_grid_points);
    cube << 0, cell_size, 0, cell_size, 0, cell_size, 0, cell_size,
        0, 0, cell_size, cell_size, 0, 0, cell_size, cell_size,
        0, 0, 0, 0, -cell_size, -cell_size, -cell_size, -cell_size;
    return cube;
}

Eigen::Matrix2Xd distorted_pixel(const double k1,
                                 const double k2,
                                 const Eigen::Vector2d &principal_pt,
                                 const Eigen::Matrix2Xd &points)
{
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

cv::Mat load_image_color(std::string image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    // std::cout << img.dims << std::endl;
    return img;
}

void
draw_circles(cv::Mat &src_img, const Eigen::Matrix2Xd &pts, int thinkness, const cv::Scalar &color, int lineType)
{
    int num_pts = pts.cols();
    for (size_t pt_idx = 0; pt_idx < num_pts; pt_idx++)
    {
        cv::circle(src_img, cv::Point2d(pts(0, pt_idx), pts(1, pt_idx)), thinkness, color, lineType);
    }
}

void draw_cube(cv::Mat &src_img, const Eigen::Matrix2Xd &pts)
{
    std::vector<std::vector<int>> edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 5}, {4, 6}, {6, 7}, {7, 5}};
    for (auto &edge : edges)
    {
        cv::line(src_img,
                 cv::Point(pts(0, edge[0]), pts(1, edge[0])),
                 cv::Point(pts(0, edge[1]), pts(1, edge[1])),
                 cv::Scalar(0, 0, 255),
                 5);
    }
}

cv::Mat undistort_image(const cv::Mat &src_img,
                        double d1,
                        double d2,
                        const Eigen::Vector2d &principal_pt)
{
    cv::Mat res = src_img.clone();
    double u0 = principal_pt(0);
    double v0 = principal_pt(1);
    for (size_t v = 0; v < src_img.rows; v++)
    {
        for (size_t u = 0; u < src_img.cols; u++)
        {
            double r_2 = (u - u0) * (u - u0) + (v - v0) * (v - v0);
            double c = 1 + d1 * r_2 + d2 * r_2 * r_2;
            int u_d = c * (u - u0) + u0;
            int v_d = c * (v - v0) + v0;
            if (u_d >= 0 && u_d < src_img.cols && v_d >= 0 && v_d < src_img.rows)
            {
                auto temp = src_img.at<cv::Vec3b>(v_d, u_d);
                res.at<cv::Vec3b>(v, u) = temp;
            }
            else
            {
                res.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return res;
}

cv::VideoWriter create_video_writer(const cv::Size &img_size, const std::string &file_path)
{
    cv::VideoWriter vid_writer;
    std::string out_data_root = "../../output/";
    auto output_pth = out_data_root + file_path;
    int fps = 30;
    vid_writer.open(output_pth, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, img_size, true);
    if (!vid_writer.isOpened())
    {
        std::stringstream err_msg;
        err_msg << "Could not open the output video for write: " << output_pth << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    return vid_writer;
}

cv::Mat convet_to_cv_to_show(const Eigen::MatrixXd& eigen_img)
{
    cv::Mat img_cv;
    cv::Mat img_cv_uchar;
    cv::eigen2cv(eigen_img, img_cv);
    cv::normalize(img_cv,img_cv_uchar , 255,0, cv::NORM_MINMAX, 0);
    return img_cv_uchar;
}