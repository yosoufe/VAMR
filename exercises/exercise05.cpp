#include "stereo_reconst.hpp"
#include "folder_manager.hpp"

int main()
{
    std::cout << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "== Exercise 05 - Stereo Dense Reconstruction ==" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << std::endl;


    std::string in_data_root{"../../data/ex05/"};

    double rescale_factor{0.5}; // Rescaling of the original image for speed.

    auto K = read_K_matrix(in_data_root + "K.txt");
    std::cout << "K matrix: \n"
              << K << "\n"
              << std::endl;

    K.block(0,0,2,3) *= rescale_factor;

    Eigen::MatrixXd R_C_frame(3, 3);
    R_C_frame << 0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        1.0, 0.0, 0.0;

    Eigen::MatrixXd R_C_frame_inverse = R_C_frame.transpose();

    auto poses = read_matrix(in_data_root + "poses.txt");
    // std::cout << "poses matrix: \n"
    //           << poses << "\n"
    //           << std::endl;

    // Given by KITTI dataset
    double baseline{0.54};

    // Carefully tuned by the TAs:
    size_t patch_radius{5};
    size_t min_disp{5};
    size_t max_disp{50};

    std::vector<double> xlims = {7, 20},
                        ylims = {-6, 10},
                        zlims = {-5, 5};

    SortedImageFiles image_files_left(in_data_root + "left/");
    SortedImageFiles image_files_right(in_data_root + "right/");

    assert(image_files_left.size() == image_files_right.size());
    assert(image_files_left.size() == poses.rows());

    std::vector<Eigen::MatrixXd> point_clouds;
    point_clouds.reserve(image_files_left.size());

    for (size_t idx = 0; idx < image_files_left.size(); ++idx)
    {
        std::cout << "frame " << idx << std::endl;
        cv::Mat left_img, right_img;
        std::string image_file_left = image_files_left[idx].path();
        std::string image_file_right = image_files_right[idx].path();

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

        // visualize_point_cloud(point_cloud);

        Eigen::MatrixXd pose = poses.block(idx, 0, 1, 12);
        pose.resize(4, 3);
        pose.transposeInPlace();
        Eigen::MatrixXd rot = pose.block(0, 0, 3, 3);
        Eigen::VectorXd translation = pose.block(0, 3, 3, 1);

        auto filtered_pc = filter_point_cloud(xlims, ylims, zlims, R_C_frame_inverse, point_cloud);

        Eigen::MatrixXd rotated_pc(filtered_pc.rows(), filtered_pc.cols());
        rotated_pc.block(0, 0, 3, rotated_pc.cols()) =
            (rot * filtered_pc.block(0, 0, 3, filtered_pc.cols())).colwise() + translation;
        rotated_pc.block(3, 0, 1, rotated_pc.cols()) = filtered_pc.block(3, 0, 1, filtered_pc.cols());

        point_clouds.push_back(rotated_pc);
    }

    visualize_point_clouds(point_clouds);

    return 0;
}