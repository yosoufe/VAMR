#include "folder_manager.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

// **************** ImageFile ****************
ImageFile::ImageFile(fs::path path, int number) : m_path(path), m_image_number(number)
{
}

string ImageFile::path() const
{
    return m_path.string();
}

int ImageFile::number() const
{
    return m_image_number;
}

// **************** SortedFiles ****************
SortedImageFiles::SortedImageFiles(std::string folder_path) : m_folder_path(folder_path)
{
    for (auto &file_ptr : fs::directory_iterator(m_folder_path))
    {
        fs::path file = file_ptr.path();
        string stem = file.stem();
        int number = stoi(stem);
        m_files.push_back(ImageFile(file, number));
    }
    sort(m_files.begin(), m_files.end(), ImageFile::comperator());
    // test
    // for (auto & file : m_files){
    //     std::cout << file.number() << endl;
    // }
}

SortedImageFiles::FileIterator::FileIterator(ImageFile *ptr) : m_ptr(ptr) {}

SortedImageFiles::FileIterator SortedImageFiles::begin()
{
    return FileIterator(&(*m_files.begin()));
}

SortedImageFiles::FileIterator SortedImageFiles::end()
{
    return FileIterator(&(*m_files.end()));
}

SortedImageFiles::FileIterator SortedImageFiles::FileIterator::operator++()
{
    return ++m_ptr;
}

bool SortedImageFiles::FileIterator::operator!=(FileIterator other)
{
    return this->m_ptr != other.m_ptr;
}

ImageFile &SortedImageFiles::FileIterator::operator*()
{
    return *m_ptr;
}

size_t SortedImageFiles::size() const
{
    return m_files.size();
}
