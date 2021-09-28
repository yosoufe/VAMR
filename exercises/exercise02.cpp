#include "utils.hpp"

auto read_detected_corners(std::string file_path)
{
    auto corners_matrix = read_matrix(file_path);
    std::vector<Eigen::Matrix2Xd> corners_in_frames;

    size_t num_points = corners_matrix.cols() / 2;

    for (auto frame : corners_matrix.rowwise())
    {
        Eigen::Matrix2Xd uvs(2, num_points);
        for (size_t idx = 0; idx < num_points; idx++)
        {
            uvs(0, idx) = frame(2 * idx);
            uvs(1, idx) = frame(2 * idx + 1);
        }
        corners_in_frames.push_back(uvs);
    }
    return corners_in_frames;
}

auto create_Q()
{
    //TODO
}

auto estimate_pose_dlt(
    const std::vector<Eigen::Matrix2Xd>& detected_corners,
    const Eigen::MatrixXd& points_in_W,
    const Eigen::Matrix3d& K)
{
    // multiply the (u, v)s by inverse of K

    // Form the question into Q. M = 0

    // Calculate the SVD of Q = U S V^T

    // Choose M as the last column of V
    // (column with smallest singular value)
    // minimize norm ||Q.M|| subject to constraint
    // ||M|| = 1

    // check if M_34 is positive,
    // otherwise multiply M by -1

    // Apply svd to R = U S V^T
    // chose R_tilde as R_tilde = U V^T
    // Orthogonal Procrustes

    // Recover teh scale of projection matrix
    // alpha = ||R_tilde|| / ||R||

    // finally M_tilde = [R_tilde | alpha . t]

    // optional: check if R orthogonal i.e. R * R^T = I
    // and det(R) = 1
}


auto re_project_points(
    const Eigen::MatrixXd& points_in_W,
    const std::vector<Eigen::MatrixXd>& Ms,
    const Eigen::Matrix3d& K)
{

}

auto plot_trajectory_3d()
{

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

    estimate_pose_dlt(detected_corners, points_in_W, K);

    return 0;
}