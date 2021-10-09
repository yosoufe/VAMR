#include <string>
#include <vector>
#include "utils.hpp"
#include "folder_manager.hpp"

cv::Mat imrotate(const cv::Mat &src, double angle_deg)
{
    cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle_deg).boundingRect2f();
    rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

    cv::Mat res;
    cv::warpAffine(src, res, rot, bbox.size());
    return res;
}

void show(const cv::Mat &img, std::string window_name = "image")
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, img);
    cv::waitKey(0);
}

Eigen::MatrixXd gaussian_vector(double sigma)
{
    size_t radius = std::ceil(3.0 * sigma);
    Eigen::MatrixXd kernel(2 * radius + 1, 1);
    for (int x = 0; x <= radius; x++)
    {
        double cst = 1.0 / (2.0 * M_PI * sigma * sigma);
        double exponent = (-1.0) * ((x * x) / (2 * sigma * sigma));
        double val = cst * std::exp(exponent);
        kernel(radius + x, 0) = val;
        kernel(radius - x, 0) = val;
    }
    kernel = kernel / kernel.sum();
    return kernel;
}

Eigen::MatrixXd gaussian_blur(const Eigen::MatrixXd &src, double sigma)
{
    auto kernel = gaussian_vector(sigma);
    Eigen::MatrixXd temp = correlation(src, kernel);
    Eigen::MatrixXd blured = correlation(temp, kernel.transpose());
    return blured;
}

int main()
{
    bool rotation_inv = false;
    double rotation_img2_deg = 0;

    size_t num_scales = 3;  // Scales per octave.
    size_t num_octaves = 5; // Number of octaves.
    double sigma = 1.6;
    double contrast_threshold = 0.04;
    std::string image_file_1 = "../../data/ex04/img_1.jpg";
    std::string image_file_2 = "../../data/ex04/img_2.jpg";
    double rescale_factor = 0.2; // Rescaling of the original image for speed.

    cv::Mat left_img = cv::imread(image_file_1, cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread(image_file_2, cv::IMREAD_GRAYSCALE);

    if (rotation_img2_deg != 0)
    {
        right_img = imrotate(right_img, rotation_img2_deg);
    }

    std::vector<cv::Mat> images = {left_img, right_img};

    for (auto &img : images)
    {
        // Write code to compute:
        // 1)    image pyramid. Number of images in the pyramid equals 'num_octaves'.
        std::vector<Eigen::MatrixXd> DoGs;

        for (size_t octave = 0; octave < num_octaves; octave++)
        {
            // 2)    blurred images for each octave. Each octave contains
            //       'num_scales + 3' blurred images.
            cv::Mat octave_img;
            double scale = 1.0 / std::pow(2, octave);
            cv::resize(img, octave_img, cv::Size(), scale, scale);

            // convert OpenCV image to Eigen Matrix
            Eigen::MatrixXd eigen_octave_img = cv_2_eigen(octave_img);
            Eigen::MatrixXd blured_down;
            for (int scale = -1; scale < int(num_scales + 1); scale++)
            {
                if (blured_down.size() == 0)
                {
                    blured_down = gaussian_blur(eigen_octave_img, std::pow(2.0, double(scale) / num_scales));
                }
                Eigen::MatrixXd blured_up = gaussian_blur(eigen_octave_img,
                                                          std::pow(2.0, double(scale + 1) / num_scales));
                // 3)    'num_scales + 2' difference of Gaussians for each octave.
                Eigen::MatrixXd DoG = blured_up - blured_down;
                DoGs.push_back(DoG);
                blured_down = blured_up;
            }
        }

        // 4)    Compute the keypoints with non-maximum suppression and
        //       discard candidates with the contrast threshold.

        // 5)    Given the blurred images and keypoints, compute the
        //       descriptors. Discard keypoints/descriptors that are too close
        //       to the boundary of the image. Hence, you will most likely
        //       lose some keypoints that you have computed earlier.
    }

    // Finally, match the descriptors using the function 'matchFeatures' and
    // visualize the matches with the function 'showMatchedFeatures'.
    // If you want, you can also implement the matching procedure yourself using
    // 'knnsearch'.

    return 0;
}