#include "utils.hpp"

auto read_detected_corners(std::string file_path)
{
    auto corners_matrix = read_matrix(file_path);
    std::vector <Eigen::Matrix2Xd> corners_in_frames;

    size_t num_points = corners_matrix.cols()/2;
    
    for (auto frame : corners_matrix.rowwise())
    {
        Eigen::Matrix2Xd uvs(2, num_points);
        for (size_t idx=0; idx<num_points; idx++)
        {
            uvs(0,idx) = frame(2*idx);
            uvs(1,idx) = frame(2*idx+1);
        }
        corners_in_frames.push_back(uvs);
    }
    return corners_in_frames;
}

auto create_Q()
{
    //TODO
}


int main(int argc, char **argv)
{
    std::string in_data_root = "../../data/ex02/";

    auto K = read_K_matrix(in_data_root + "K.txt");
    std::cout << "K matrix: \n"
              << K << "\n"
              << std::endl;

    auto points_in_W = read_matrix(in_data_root + "p_W_corners.txt", ',').transpose();
    std::cout << "Points in world coordinate: \n"
              << points_in_W << "\n"
              << std::endl;

    auto detected_corners = read_detected_corners(in_data_root + "detected_corners.txt");

    return 0;
}