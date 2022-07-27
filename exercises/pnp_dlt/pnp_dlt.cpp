#include "pnp_dlt.hpp"

#include <opencv2/viz.hpp>
#include <chrono>
#include <thread>
#include "utils.hpp"
#include "camera_model.hpp"

std::vector<Eigen::Matrix3Xd>
read_detected_corners(std::string file_path)
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
            uvs(2, idx) = 1.0;
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
    if (M.block(0, 0, 3, 3).determinant() < 0.0)
    {
        M *= -1.0;
    }
    // std::cout << "M:" << std::endl << M << std::endl;

    // Apply svd to R = U S V^T
    // chose R_tilde as R_tilde = U V^T
    // Orthogonal Procrustes
    Eigen::MatrixXd R = M.block(0, 0, 3, 3);
    Eigen::BDCSVD<Eigen::MatrixXd> svd_r(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::MatrixXd R_tilde = svd_r.matrixU() * (svd_r.matrixV().transpose());
    // std::cout << "R_tilde:" << std::endl << R_tilde << std::endl;

    // Recover the scale of projection matrix
    // alpha = ||R_tilde|| / ||R||
    double alpha = R_tilde.norm() / R.norm();
    // std::cout << "alpha:" << std::endl << alpha << std::endl;

    // finally M_tilde = [R_tilde | alpha . t]
    Eigen::MatrixXd M_tilde = M;
    M_tilde.block(0, 0, 3, 3) = R_tilde;
    M_tilde.block(0, 3, 3, 1) = alpha * M.block(0, 3, 3, 1);
    // std::cout << "M_tilde:" << std::endl
    //           << M_tilde << std::endl;

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
    return project_2_camera_frame(K, extrinsics, points_in_W);
}

void print_current_pose_of_camera(const cv::viz::Viz3d &window)
{
    std::cout << "translation: " << window.getViewerPose().translation() << std::endl;
    std::cout << "rotation: " << window.getViewerPose().rotation() << std::endl;
}

void plot_trajectory_3d(
    const std::vector<Eigen::MatrixXd> &poses,
    const std::vector<cv::Mat> &images,
    const Eigen::MatrixXd &points_in_W,
    const std::string &output_file)
{
    using namespace cv;
    auto img_size = images[0].size();
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setWindowSize(Size(img_size.width * 2, img_size.height));
    myWindow.setWindowPosition(Point(2560, 0));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(0.1));

    myWindow.setViewerPose(
        Affine3d(
            Matx33d(
                0.4700092538834332, -0.2668985226622341, 0.8413420706613068,
                -0.1346380279134217, 0.9203547147052047, 0.3671781591536814,
                -0.8723324496449293, -0.2858537698205474, 0.3966405420224444),
            Vec3d(-0.581886, -0.289294, -0.987273)));

    for (size_t pt_idx = 0; pt_idx < points_in_W.cols(); pt_idx++)
    {
        auto sphere_w = viz::WSphere(Point3d(points_in_W(0, pt_idx) / 100.0,
                                             points_in_W(1, pt_idx) / 100.0,
                                             points_in_W(2, pt_idx) / 100.0),
                                     0.01, 4, viz::Color::red());
        std::stringstream str_;
        str_ << "sphere_" << pt_idx;
        myWindow.showWidget(str_.str(), sphere_w);
    }

    viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599), 0.1); // Camera frustum
    viz::WCameraPosition cpw(.05);                                    // Coordinate axes

    cv::VideoWriter re_projected_vid_writer = create_video_writer(Size(img_size.width * 2, img_size.height), output_file);

    int counter = 0;
    bool first_time = false;
    myWindow.spinOnce(1000, true);

    for (size_t counter = 0; counter < poses.size() && !myWindow.wasStopped(); counter++)
    {
        auto img_w = viz::WImageOverlay(images[counter],
                                        Rect(img_size.width,
                                             0,
                                             img_size.width,
                                             img_size.height));
        myWindow.showWidget("Image", img_w);

        Eigen::MatrixXd cam_ei_rot = poses[counter].block(0, 0, 3, 3);
        cam_ei_rot.transposeInPlace();
        Eigen::MatrixXd cam_ei_tra = poses[counter].block(0, 3, 3, 1);
        cam_ei_tra = (-1 * cam_ei_rot) * cam_ei_tra;

        cam_ei_rot.transposeInPlace();
        cam_ei_tra.transposeInPlace();
        cam_ei_tra /= 100;
        Mat cam_rot(3, 3, CV_64FC1, cam_ei_rot.data());
        Mat cam_tra(3, 1, CV_64FC1, cam_ei_tra.data());
        Affine3d cam_pose(
            cam_rot,
            cam_tra);

        myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
        myWindow.showWidget("CPW", cpw, cam_pose);
        if (first_time)
        {
            myWindow.spinOnce(1000);
            first_time = false;
            print_current_pose_of_camera(myWindow);
        }        
        re_projected_vid_writer << myWindow.getScreenshot();
        myWindow.spinOnce(1, true);
    }
    myWindow.close();
}