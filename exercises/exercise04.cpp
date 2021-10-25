#include <string>
#include <vector>
#include "sift.hpp"
#include "folder_manager.hpp"

// DELETE, OPENCV SIFT Detector
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

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

int main()
{
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "======= Exercise 04 - SIFT =======" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;

    bool rotation_inv = false;
    double rotation_img2_deg = 0;

    size_t num_scales_in_octave = 3;  // Scales per octave.
    size_t num_octaves = 5; // Number of octaves.
    double sigma = 1.6;
    double contrast_threshold = 0.4;
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
        std::vector<Eigen::MatrixXd> DoGs;
        // Write code to compute:
        // 1)    image pyramid. Number of images in the pyramid equals 'num_octaves'.

        for (size_t octave = 0; octave < num_octaves; octave++)
        {
            cv::Mat octave_img;
            double scale = 1.0 / std::pow(2, octave);
            cv::resize(img, octave_img, cv::Size(), scale, scale, cv::INTER_CUBIC);

            // convert OpenCV image to Eigen Matrix
            Eigen::MatrixXd eigen_octave_img = cv_2_eigen(octave_img);

            // 3) calculate DOGs
            calculate_DoGs(num_scales_in_octave, eigen_octave_img, DoGs, sigma);
        }
        //// opencv SIFT keypoints, uncomment the headers on the top if
        //// required for testing

        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(img, keypoints);
        cv::Mat output;
        cv::drawKeypoints(img, keypoints, output);
        std::cout << "number of cv keypoints " << keypoints.size() << std::endl;
        show(output, "CV SIFT");

        // 4)    Compute the keypoints with non-maximum suppression and
        //       discard candidates with the contrast threshold.
        auto kpts = find_keypoints(DoGs,
                                   contrast_threshold,
                                   num_octaves);

        std::cout << "number of keypoints " << kpts.cols() << std::endl;

        // 5)    Given the blurred images and keypoints, compute the
        //       descriptors. Discard keypoints/descriptors that are too close
        //       to the boundary of the image. Hence, you will most likely
        //       lose some keypoints that you have computed earlier.

        show_kpts_in_images(kpts, img, num_octaves, num_scales_in_octave);

        break;
    }

    // Finally, match the descriptors using the function 'matchFeatures' and
    // visualize the matches with the function 'showMatchedFeatures'.
    // If you want, you can also implement the matching procedure yourself using
    // 'knnsearch'.

    return 0;
}