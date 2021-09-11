#include <iostream>
#include <fstream>      // ifstream
#include <Eigen/Core>
#include <Eigen/Geometry> // AngleAxis
#include <Eigen/Dense>
#include <vector>
#include <cassert>

std::ifstream read_file(std::string path){
    std::ifstream fin(path);
    if (!fin || !fin.is_open()){
        std::stringstream error_msg;
        error_msg << "cannot find pose file at " << path << std::endl;
        std::cout << error_msg.str();
    }
    fin.seekg(0, std::ios::beg);
    return fin;
}

auto read_pose_file(std::string file_path){
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    
    auto fin = read_file(file_path);
    if (!fin){
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

void read_distortion_param(std::string file_path, double& k1, double& k2){
    auto fin = read_file(file_path);
    if (!fin){
        return;
    }

    fin >> k1 >> k2;
    return;
}

auto read_K_matrix(std::string file_path){
    auto fin = read_file(file_path);
    if (!fin){
        return Eigen::Matrix3d();
    }
    std::vector<double> buf;
    size_t counter = 0;
    while (!fin.eof() && counter < 9) {
        double v = 0;
        fin >> v;
        buf.push_back(v);
        counter++;
    }
    Eigen::Matrix3d k_matrix(buf.data());
    k_matrix.resize(3,3);
    return Eigen::Matrix3d(k_matrix.transpose());
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
    auto poses = read_pose_file("../../data/ex01/poses.txt");
    if (poses.size()==0){
        return -1;
    }
    auto grid = create_grid(0.04, 9, 6);
    // std::cout << grid << std::endl;
    auto K = read_K_matrix("../../data/ex01/K.txt");
    std::cout << K << std::endl;
    double d1 = 0, d2=0;
    read_distortion_param("../../data/ex01/D.txt", d1, d2);
    std::cout << d1 << " " << d2 << std::endl;
}