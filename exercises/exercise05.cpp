#include "stereo_reconst.hpp"

int main()
{
    std::cout << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "== Exercise 05 - Stereo Dense Reconstruction ==" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << std::endl;

    std::string in_data_root{"../../data/ex05/"};

    auto K = read_K_matrix(in_data_root + "K.txt");
    std::cout << "K matrix: \n"
              << K << "\n"
              << std::endl;

    auto poses = read_matrix(in_data_root + "poses.txt");
    // std::cout << "poses matrix: \n"
    //           << poses << "\n"
    //           << std::endl;

    cv::Mat left_img, right_img;
    std::string image_file_left = in_data_root + "left/000000.png";
    std::string image_file_right = in_data_root + "right/000000.png";

    double rescale_factor{0.5}; // Rescaling of the original image for speed.
    cv::resize(cv::imread(image_file_left,
                          cv::IMREAD_GRAYSCALE),
               left_img,
               cv::Size(),
               rescale_factor,
               rescale_factor,
               cv::INTER_CUBIC);
    cv::resize(cv::imread(image_file_right,
                          cv::IMREAD_GRAYSCALE),
               right_img,
               cv::Size(),
               rescale_factor,
               rescale_factor,
               cv::INTER_CUBIC);

    // Given by KITTI dataset
    double baseline{0.54};

    // Carefully tuned by the TAs:
    size_t patch_radius{5};
    size_t min_disp{5};
    size_t max_disp{50};
    std::vector<double> xlims = {7, 20},
                        ylims = {-6, 10},
                        zlims = {-5, 5};

    auto left_eigen = cv_2_eigen(left_img);
    auto right_eigen = cv_2_eigen(right_img);

    auto disp_img = get_disparity(left_eigen,
                                  right_eigen,
                                  patch_radius,
                                  min_disp,
                                  max_disp);

    // show(eigen_2_cv(disp_img * 255 / max_disp));

    auto point_cloud =
        disparity_to_pointcloud(disp_img,
                                K,
                                baseline,
                                left_eigen);

    visualize_point_cloud(point_cloud);

    return 0;
}