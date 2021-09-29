#include "utils.hpp"

auto read_detected_corners(std::string file_path)
{
    auto corners_matrix = read_matrix(file_path);
    std::vector<Eigen::Matrix3Xd> corners_in_frames;

    size_t num_points = corners_matrix.cols() / 2;

    for (auto frame : corners_matrix.rowwise())
    {
        Eigen::Matrix3Xd uvs(3, num_points);
        for (size_t idx = 0; idx < num_points; idx++)
        {
            uvs(0, idx) = frame(2 * idx);
            uvs(1, idx) = frame(2 * idx + 1);
            uvs(2, idx) = 1.0d;
        }
        corners_in_frames.push_back(uvs);
    }
    return corners_in_frames;
}

auto create_Q(
    const Eigen::Matrix3Xd &normalized_corners,
    const Eigen::MatrixXd &points_in_W)
{
    size_t num_corners = normalized_corners.cols();
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(2 * num_corners, 12);
    for (size_t corner_idx = 0; corner_idx < num_corners; corner_idx++)
    {
        // std::cout  << "res:" << std::endl << res << std::endl;
        double X_w = points_in_W(0, corner_idx);
        double Y_w = points_in_W(1, corner_idx);
        double Z_w = points_in_W(2, corner_idx);

        double x = normalized_corners(0, corner_idx);
        double y = normalized_corners(1, corner_idx);

        res.block(2 * corner_idx, 0, 1, 4) << X_w, Y_w, Z_w, 1.0;
        res.block(2 * corner_idx + 1, 4, 1, 4) << X_w, Y_w, Z_w, 1.0;
        res.block(2 * corner_idx, 8, 2, 4) << -x * X_w, -x * Y_w, -x * Z_w, -x,
            -y * X_w, -y * Y_w, -y * Z_w, -y;
    }
    return res;
}

Eigen::MatrixXd estimate_pose_dlt(
    const Eigen::Matrix3Xd &detected_corners,
    const Eigen::MatrixXd &points_in_W,
    const Eigen::Matrix3d &K)
{
    // multiply the (u, v)s by inverse of K
    Eigen::Matrix3d K_inv = K.inverse();
    // K_inv is 3X3
    // detected_corners is 3 x number_of_corners
    // print_shape(K_inv);
    // print_shape(detected_corners);
    Eigen::Matrix3Xd normalized_corners = K_inv * detected_corners;

    // Form the question into Q. M = 0
    Eigen::MatrixXd Q = create_Q(normalized_corners, points_in_W);

    // Calculate the SVD of Q = U S V^T
    // JacobiSVD vs BDCSVD
    Eigen::BDCSVD<Eigen::MatrixXd> svd_m(Q, Eigen::ComputeFullV);

    // Choose M as the last column of V
    // (column with smallest singular value)
    // minimize norm ||Q.M|| subject to constraint
    // ||M|| = 1
    Eigen::MatrixXd M = Eigen::MatrixXd(svd_m.matrixV().col(11));

    // std::cout << "M:" << std::endl << M << std::endl;
    M.resize(4, 3);
    M.transposeInPlace();
    // std::cout << "M resized:" << std::endl << M << std::endl;

    // check if M_34 is positive,
    // otherwise multiply M by -1
    if (M(2, 3) < 0.0)
    {
        M *= -1.0;
    }
    // std::cout << "M:" << std::endl << M << std::endl;

    // Apply svd to R = U S V^T
    // chose R_tilde as R_tilde = U V^T
    // Orthogonal Procrustes
    Eigen::MatrixXd R = M.block(0, 0, 3, 3);
    Eigen::BDCSVD<Eigen::MatrixXd> svd_r(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::MatrixXd R_tilde = svd_r.matrixU() * svd_r.matrixV().transpose();
    // std::cout << "R_tilde:" << std::endl << R_tilde << std::endl;

    // Recover the scale of projection matrix
    // alpha = ||R_tilde|| / ||R||
    double alpha = R_tilde.norm() / R.norm();
    // std::cout << "alpha:" << std::endl << alpha << std::endl;

    // finally M_tilde = [R_tilde | alpha . t]
    Eigen::MatrixXd M_tilde = M;
    M_tilde.block(0, 0, 3, 3) = R_tilde;
    M_tilde.block(0, 3, 3, 1) = alpha * M.block(0, 3, 3, 1);
    // std::cout  << "M_tilde:" << std::endl << M_tilde << std::endl;

    // optional: check if R orthogonal i.e. R * R^T = I
    // and det(R) = 1

    // return the results
    return M_tilde;
}

Eigen::MatrixXd re_project_points(
    const Eigen::MatrixXd &points_in_W,
    const Eigen::MatrixXd &pose,
    const Eigen::Matrix3d &K)
{
    Eigen::Isometry3d extrinsics;
    extrinsics.matrix().block(0, 0, 3, 4) = pose;
    // std::cout  << "extrinsics:" << std::endl << extrinsics.matrix() << std::endl;
    return project_2_camera_frame(K, extrinsics, points_in_W);
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

    Eigen::MatrixXd points_in_W = read_matrix(in_data_root + "p_W_corners.txt", ',').transpose();
    std::cout << "Points in world coordinate: \n"
              << points_in_W << "\n"
              << std::endl;

    auto all_detected_corners = read_detected_corners(in_data_root + "detected_corners.txt");

    // load single image to get image size
    auto image = load_image(in_data_root + "images_undistorted/img_0001.jpg");
    auto img_size = image.size();
    cv::VideoWriter re_projected_vid_writer = create_video_writer(img_size, "ex02/reprojected_points.mp4");

    std::vector<Eigen::MatrixXd> poses;

    for (size_t image_idx = 0; image_idx < 210; image_idx++)
    {
        auto &detected_corners = all_detected_corners[image_idx];

        std::stringstream image_path;
        image_path << "images_undistorted/img_" << std::setfill('0') << std::setw(4) << image_idx+1 << ".jpg";
        image = load_image(in_data_root + image_path.str());

        draw_circles(image, detected_corners.block(0,0,2,detected_corners.cols()), 3, cv::Scalar(0, 0, 255), cv::LINE_8);

        Eigen::MatrixXd pose = estimate_pose_dlt(detected_corners, points_in_W, K);
        Eigen::MatrixXd re_projected_points = re_project_points(points_in_W, pose, K);
        std::cout << "differences between original and re-projected:" << std::endl
                  << re_projected_points - detected_corners.block(0, 0, 2, 12) << std::endl;
        poses.push_back(pose);

        draw_circles(image, re_projected_points, 2, cv::Scalar(255, 0, 0), cv::LINE_4);

        re_projected_vid_writer << image;
    }

    return 0;
}