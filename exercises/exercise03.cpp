#include "utils.hpp"
#include "folder_manager.hpp"


int main(){
    std::string in_data_root = "../../data/ex03/";
    SortedImageFiles files(in_data_root);

    for (auto & image_path : files)
    {
        std::cout << image_path.path() << std::endl;
    }
    return 0;
}