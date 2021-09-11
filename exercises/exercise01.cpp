#include <iostream>
#include <fstream>      // ifstream
#include <Eigen/Core>
#include <Eigen/Geometry> // AngleAxis
#include <Eigen/Dense>
#include <vector>


std::string poses_path = "../../data/ex01/poses.txt";

auto read_pose_file(std::string file_path){
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    
    std::ifstream fin(poses_path);
    if (!fin){
        std::stringstream error_msg;
        error_msg << "cannot find pose file at " << poses_path << std::endl;
        std::cout << error_msg.str();
        return poses;
    }
    
    // read pose file
    while (!fin.eof()) {
        double wx, wy, wz, tx, ty, tz;
        fin >> wx >> wy >> wz >> tx >> ty >> tz;
        Eigen::Vector3d rot(wx,wy,wz);
        double angle = rot.norm();
        Eigen::AngleAxisd rot_angleaxis(angle, rot/angle);
        
        Eigen::Isometry3d Twr(rot_angleaxis);
        Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
        poses.push_back(Twr);
    }
    return poses;
}

Eigen::MatrixXd create_grid(double cell_size, size_t num_x, size_t num_y){
    size_t num_grid_points = num_x * num_y;
    Eigen::MatrixXd grid;
    grid.resize(3,num_grid_points);
    for (size_t idx=0 ; idx < num_grid_points; idx++){
        double x = idx % num_x;
        double y = idx / num_x;
        grid.block(0,idx,3,1) << x*cell_size, y*cell_size, 0.0d;
    }
    return grid;
}

int main(int argc, char **argv){
    auto poses = read_pose_file(poses_path);
    if (poses.size()==0){
        return -1;
    }
    auto grid = create_grid(0.04, 9, 6);
    std::cout << grid << std::endl;
}