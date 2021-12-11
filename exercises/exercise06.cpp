#include "utils.hpp"
#include "two_view_geometry.hpp"
#include <opencv2/viz.hpp>

void visualize_point_cloud(
    Eigen::MatrixXd const &point_cloud,
    Eigen::MatrixXd const &R_C2_W,
    Eigen::MatrixXd const &T_C2_W);

int main()
{
    std::cout << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "== Exercise 06 - Two-view Geometry ==" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;

    std::string in_data_root{"../../data/ex06/"};

    cv::Mat img = cv::imread(in_data_root + "0001.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(in_data_root + "0002.jpg", cv::IMREAD_GRAYSCALE);

    Eigen::MatrixXd K(3, 3);
    K << 1379.74, 0, 760.35,
        0, 1382.08, 503.41,
        0, 0, 1;

    // Load outlier-free point correspondences
    Eigen::MatrixXd p1 = read_matrix(in_data_root + "matches0001.txt");
    Eigen::MatrixXd p2 = read_matrix(in_data_root + "matches0002.txt");

    p1.conservativeResize(3, Eigen::NoChange_t::NoChange);
    p1.row(2) = Eigen::MatrixXd::Ones(1, p1.cols());

    p2.conservativeResize(3, Eigen::NoChange_t::NoChange);
    p2.row(2) = Eigen::MatrixXd::Ones(1, p2.cols());

    // Estimate the essential matrix E using the 8-point algorithm
    Eigen::MatrixXd E = estimate_essential_matrix(p1, p2, K, K);

    // Extract the relative camera positions (R,T) from the essential matrix

    // Obtain extrinsic parameters (R,t) from E
    std::vector<Eigen::MatrixXd> Rots;
    Eigen::MatrixXd u3;
    decompose_essential_matrix(E, Rots, u3);

    Eigen::MatrixXd R_C2_W;
    Eigen::MatrixXd T_C2_W;
    disambiguate_relative_pose(
        Rots,
        u3,
        p1,
        p2,
        K,
        K,
        R_C2_W,
        T_C2_W);

    std::cout << "\nR_C2_W=\n" << R_C2_W << std::endl;
    std::cout << "\nT_C2_W=\n" << T_C2_W << std::endl;

    // Triangulate a point cloud using the final transformation (R,T)
    Eigen::MatrixXd M1 = K * Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd M2(3, 4);
    // I do not know why I need to reverse the transformation here
    // TODO: discover why???
    M2 << K * R_C2_W.transpose(), (-1) * K * R_C2_W.transpose() * T_C2_W;
    Eigen::MatrixXd P = linear_triangulation(p1, p2, M1, M2);

    visualize_point_cloud(P, R_C2_W, T_C2_W);

    return 0;
}

void viz_camera(
    Eigen::MatrixXd const &cam_R,
    Eigen::MatrixXd const &cam_T,
    cv::viz::Viz3d &widget,
    std::string cam_name)
{
    using namespace cv;
    Eigen::MatrixXd cam_ei_rot = cam_R;
    cam_ei_rot.transposeInPlace();
    Eigen::MatrixXd cam_ei_tra = cam_T;

    Mat cam_rot(3, 3, CV_64FC1, cam_ei_rot.data());
    Mat cam_tra(3, 1, CV_64FC1, cam_ei_tra.data());
    viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599), 1.0); // Camera frustum
    viz::WCameraPosition cpw(0.5);                                    // Coordinate axes
    cv::Affine3d cam_pose(cam_rot, cam_tra);

    widget.showWidget(cam_name + "_CPW_FRUSTUM" , cpw_frustum, cam_pose);
    widget.showWidget(cam_name + "_CPW", cpw, cam_pose);
}

void visualize_point_cloud(
    Eigen::MatrixXd const &point_cloud,
    Eigen::MatrixXd const &R_C2_W,
    Eigen::MatrixXd const &T_C2_W)
{
    using namespace cv;
    viz::Viz3d myWindow("Point Cloud");
    myWindow.setWindowSize(Size(1920, 1080));
    // location of the window,
    // Comment it out if you have only single monitor
    myWindow.setWindowPosition(Point(2560, 0));

    cv::Mat cv_cloud;
    Eigen::MatrixXd pc_transpose = point_cloud.transpose().block(0,0,point_cloud.cols(), 3);
    cv::eigen2cv(pc_transpose, cv_cloud);
    cv_cloud = cv_cloud.reshape(3);

    // https://answers.opencv.org/question/65569/cvviz-point-cloud/
    cv::viz::WCloud cloud_widget{cv_cloud, cv::viz::Color::red()}; // cv_color
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 4);

    std::stringstream str_;
    str_ << "point_cloud";
    myWindow.showWidget(str_.str(), cloud_widget);

    myWindow.spinOnce(1, true);
    myWindow.setViewerPose(
        Affine3d(
            Matx33d(
                1, 0, 0,
                0, 0, 1,
                0, -1, 0),
            Vec3d(0.295923, -2.4547670, .4077167)));

    viz_camera(Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,1), myWindow, "cam1");
    viz_camera(R_C2_W, T_C2_W, myWindow, "cam2");

    myWindow.spin();
    myWindow.close();
}