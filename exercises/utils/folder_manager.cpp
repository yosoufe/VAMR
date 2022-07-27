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
        int number = stoi(first_numberstring(stem));
        m_files.push_back(ImageFile(file, number));
    }
    sort(m_files.begin(), m_files.end(), ImageFile::comperator());
}

std::string SortedImageFiles::first_numberstring(std::string const &str)
{
    char const *digits = "0123456789";
    std::size_t const n = str.find_first_of(digits);
    if (n != std::string::npos)
    {
        std::size_t const m = str.find_first_not_of(digits, n);
        return str.substr(n, m != std::string::npos ? m - n : m);
    }
    return std::string();
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

const ImageFile &SortedImageFiles::operator[](int index) const
{
    return m_files[index];
}
