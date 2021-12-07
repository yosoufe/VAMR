#include "utils.hpp"
#include "two_view_geometry.hpp"
#include <opencv2/viz.hpp>

void visualize_point_cloud(Eigen::MatrixXd const &point_cloud);

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

    // std::cout << "Rots[0] \n" << Rots[0] << std::endl;
    // std::cout << "Rots[0].determinant() \n" << Rots[0].determinant() << std::endl;
    // std::cout << "Rots[1] \n" << Rots[1] << std::endl;
    // std::cout << "Rots[1].determinant() \n" << Rots[1].determinant() << std::endl;
    // std::cout << "u3 \n" << u3 << std::endl;

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
        T_C2_W
    );

    // Triangulate a point cloud using the final transformation (R,T)
    Eigen::MatrixXd M1 = K * Eigen::MatrixXd::Identity(3,4);
    Eigen::MatrixXd M2(3,4);
    M2 << R_C2_W, T_C2_W;
    M2 = K * M2;
    Eigen::MatrixXd P = linear_triangulation(p1, p2, M1, M2);

    visualize_point_cloud(P);

    return 0;
}

void visualize_point_cloud(Eigen::MatrixXd const &point_cloud)
{
    using namespace cv;
    viz::Viz3d myWindow("Point Cloud");
    myWindow.setWindowSize(Size(1920, 1080));
    // location of the window,
    // Comment it out if you have only single monitor
    myWindow.setWindowPosition(Point(2560, 0));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(1));

    cv::Mat cv_cloud;
    Eigen::MatrixXd pc_transpose = point_cloud.transpose();
    cv::eigen2cv(pc_transpose, cv_cloud);
    cv_cloud = cv_cloud.reshape(4);

    // cv::Mat cv_color;
    // Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigen_color =
    //     point_cloud.block(3, 0, 1, point_cloud.cols()).cast<unsigned char>().transpose();
    // cv::eigen2cv(eigen_color, cv_color);

    // std::cout << "cv_cloud\n" << cv_cloud << std::endl;
    // std::cout << "point_cloud\n" << point_cloud.transpose() << std::endl;



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
            Vec3d(0.295923,-24.547670,4.077167)));

    myWindow.spin();
    myWindow.close();
}