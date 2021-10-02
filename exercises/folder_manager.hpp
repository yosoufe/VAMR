#include <string>
#include <vector>
#include <filesystem>

// **************** ImageFile ****************

class ImageFile
{
private:
    std::filesystem::path m_path;
    int m_image_number;

public:
    ImageFile(std::filesystem::path path, int number);
    std::string path() const;
    int number() const;

    struct comperator
    {
        inline bool operator()(const ImageFile &file1, const ImageFile &file2)
        {
            return (file1.number() < file2.number());
        }
    };
};

// **************** SortedFiles ****************

class SortedImageFiles
{
private:
    std::string m_folder_path;
    std::vector<ImageFile> m_files;

public:
    SortedImageFiles(std::string folder_path);

    class FileIterator
    {
    private:
        ImageFile *m_ptr;

    public:
        FileIterator(ImageFile *ptr);
        FileIterator operator++();
        bool operator!=(FileIterator);
        ImageFile &operator*();
    };
    using Iterator = FileIterator;

    FileIterator begin();
    FileIterator end();
    const ImageFile &operator[](int index) const;
    size_t size() const;
};
