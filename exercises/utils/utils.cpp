#include "utils.hpp"
#include <fstream> // ifstream

std::ifstream read_file(std::string path)
{
    std::ifstream fin(path);
    if (!fin || !fin.is_open())
    {
        std::stringstream error_msg;
        error_msg << "cannot find pose file at " << path << std::endl;
        std::cout << error_msg.str() << std::endl;
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

cv::VideoWriter create_video_writer(const cv::Size &img_size, const std::string &file_path)
{
    cv::VideoWriter vid_writer;
    int fps = 30;
    vid_writer.open(file_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, img_size, true);
    if (!vid_writer.isOpened())
    {
        std::stringstream err_msg;
        err_msg << "Could not open the output video to write: " << file_path << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    return vid_writer;
}

cv::Mat convert_to_cv_to_show(const Eigen::MatrixXd& eigen_img)
{
    cv::Mat img_cv;
    cv::Mat img_cv_uchar;
    cv::eigen2cv(eigen_img, img_cv);
    cv::normalize(img_cv,img_cv_uchar , 255,0, cv::NORM_MINMAX, 0);
    // img_cv.convertTo(img_cv_uchar, 0);
    // img_cv_uchar = img_cv;
    return img_cv_uchar;
}

Eigen::MatrixXd cv_2_eigen(const cv::Mat &img)
{
    Eigen::MatrixXd eigen_img;
    cv::cv2eigen(img, eigen_img);
    return eigen_img;
}

cv::Mat eigen_2_cv(const Eigen::MatrixXd &eigen)
{
    cv::Mat img;
    typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXuc;
    MatrixXuc temp = eigen.cast<unsigned char>();
    cv::eigen2cv(temp, img);
    return img;
}

void show(const cv::Mat &img, std::string window_name)
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, img);
    cv::resizeWindow(window_name, 1920, 1080);
    cv::waitKey(0);
}

std::string cv_type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void visualize_matrix_as_image(Eigen::MatrixXd mat)
{
    auto mat_cv = convert_to_cv_to_show(mat);
    cv::imshow("output", mat_cv);
    cv::waitKey(0);
}