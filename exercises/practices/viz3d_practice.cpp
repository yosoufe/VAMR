#if 1
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

static Mat cvcloud_load()
{
    Mat cloud(1, 1889, CV_32FC3);
    ifstream ifs("../bunny.ply");
    string str;
    for (size_t i = 0; i < 12; ++i)
        getline(ifs, str);
    Point3f *data = cloud.ptr<cv::Point3f>();
    float dummy1, dummy2;
    for (size_t i = 0; i < 1889; ++i)
        ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2;
    cloud *= 5.0f;
    return cloud;
}
int main(int argn, char **argv)
{
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setWindowSize(Size(1600, 600));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    myWindow.setWindowPosition(Point(2560, 0));
    Vec3f cam_pos(3.0f, 3.0f, 3.0f), cam_focal_point(3.0f, 3.0f, 2.0f), cam_y_dir(-1.0f, 0.0f, 0.0f);
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f, -1.0f, 0.0f), Vec3f(-1.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), cam_pos);
    Mat bunny_cloud = cvcloud_load();
    viz::WCloud cloud_widget(bunny_cloud, viz::Color::green());
    Affine3f cloud_pose = Affine3f().translate(Vec3f(0.0f, 0.0f, 3.0f));
    Affine3f cloud_pose_global = transform * cloud_pose;

    viz::WCameraPosition cpw(0.5);                               // Coordinate axes
    viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
    myWindow.showWidget("CPW", cpw, cam_pose);
    myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    myWindow.setViewerPose(
        viz::makeCameraPose(
            Vec3f(6.0, 2.5, 4.0)*2,
            Vec3f(0.0f, 2.5f, 0.0f)*2,
            Vec3f(0, 0, -1.0)));

    Mat img = imread("../ubuntu.png", IMREAD_COLOR);
    myWindow.showWidget("An Image", viz::WImageOverlay(img, Rect(800, 0, 800, 600)));

    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);

    myWindow.spin();
    return 0;
}
#else

#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
static void help()
{
    cout
        << "--------------------------------------------------------------------------" << endl
        << "This program shows how to visualize a cube rotated around (1,1,1) and shifted "
        << "using Rodrigues vector." << endl
        << "Usage:" << endl
        << "./widget_pose" << endl
        << endl;
}
int main()
{
    help();
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setWindowSize(Size(1600, 600));
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    Mat img = imread("../ubuntu.png", IMREAD_COLOR);
    myWindow.showWidget("An Image", viz::WImageOverlay(img, Rect(800, 0, 800, 600)));

    viz::WCube cube_widget(Point3f(0.5, 0.5, 0.0), Point3f(0.0, 0.0, -0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Cube Widget", cube_widget);
    Mat rot_vec = Mat::zeros(1, 3, CV_32F);
    float translation_phase = 0.0, translation = 0.0;
    rot_vec.at<float>(0, 0) += (float)CV_PI * 0.01f;
    rot_vec.at<float>(0, 1) += (float)CV_PI * 0.01f;
    rot_vec.at<float>(0, 2) += (float)CV_PI * 0.01f;
    translation_phase += (float)CV_PI * 0.01f;
    translation = sin(translation_phase);
    Mat rot_mat;
    Rodrigues(rot_vec, rot_mat);
    cout << "rot_mat = " << rot_mat << endl;
    Affine3f pose(rot_mat, Vec3f(translation, translation, translation));
    Affine3f pose2(pose.matrix);
    cout << "pose = " << pose.matrix << endl;
    cout << "pose = " << pose2.matrix << endl;
    while (!myWindow.wasStopped())
    {
        /* Rotation using rodrigues */
        rot_vec.at<float>(0, 0) += (float)CV_PI * 0.01f;
        rot_vec.at<float>(0, 1) += (float)CV_PI * 0.01f;
        rot_vec.at<float>(0, 2) += (float)CV_PI * 0.01f;
        translation_phase += (float)CV_PI * 0.01f;
        translation = sin(translation_phase);
        Mat rot_mat1;
        Rodrigues(rot_vec, rot_mat1);
        Affine3f pose1(rot_mat1, Vec3f(translation, translation, translation));
        myWindow.setWidgetPose("Cube Widget", pose1);
        myWindow.spinOnce(1, true);
    }
    return 0;
}

#endif