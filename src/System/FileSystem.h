#pragma once

#include <Windows.h>
#include "Collection.h"
#include "Type.h"

namespace TurboCV
{
namespace System
{
    // Provides the base class for both FileInfo and DirectoryInfo objects.
    class FileSystemInfo
    {
    public:
        // Initializes a new instance of the FileSystemInfo class.
        FileSystemInfo(const String& path);

        // Gets the string representing the extension part of the file.
        String Extension() const;

        // For files, gets the name of the file. For directories, gets the name of 
        // the last directory in the hierarchy if a hierarchy exists. Otherwise, 
        // the Name method gets the name of the directory.
        String Name() const;

        // Gets the full path of the directory or file.
        String FullName() const;

        // Gets a value indicating whether the file or directory exists.
        virtual bool Exists() const = 0;

        // Creates a file or directory.
        virtual bool Create() const = 0;

        // Deletes a file or directory.
        virtual bool Delete() const = 0;

        // Moves a specified file / direcotry to a new location, 
        // providing the option to specify a new file / directory name.
        bool MoveTo(const String& newPath);

    protected:
        String _path;
    };

    // Initializes a new instance of the FileSystemInfo class.
    inline FileSystemInfo::FileSystemInfo(const String& path)
    {
        size_t pathLen = path.Length();
        size_t lastPos = pathLen - 1;

        // remove all the '\\'s at the end of the path
        while (lastPos != 0 && path[lastPos] == '\\')
            lastPos--;

        if (lastPos == 0 && path[lastPos] == '\\')
            _path = "";
        else
            _path = path.Substring(0, lastPos + 1);
    }

    // Gets the string representing the extension part of the file.
    inline String FileSystemInfo::Extension() const
    {
        size_t lastBacklash = _path.LastIndexOf('\\');
        size_t lastDot = _path.LastIndexOf('.');

        if (lastDot != -1 && lastBacklash < lastDot)
            return _path.Substring(lastDot);
        else
            return "";
    }

    // For files, gets the name of the file. For directories, gets the name of 
    // the last directory in the hierarchy if a hierarchy exists. Otherwise, 
    // the Name method gets the name of the directory.
    inline String FileSystemInfo::Name() const
    {
        return _path.Substring(_path.LastIndexOf('\\') + 1);
    }

    // Gets the full path of the directory or file.
    inline String FileSystemInfo::FullName() const
    {
        TCHAR path[MAX_PATH] = TEXT(""); 
        TCHAR** fileName = { NULL };
        TCHAR curDir[MAX_PATH];
        String result;

        GetCurrentDirectory(MAX_PATH, curDir);
        SetCurrentDirectory(_path);

        if (GetFullPathName(_path, MAX_PATH, path, fileName))
        {
            if (fileName == NULL) // if it is a directory
                result = String(path);
            else // otherwise it is a file
                result = String(path) + "\\" + String(*fileName);
        }

        SetCurrentDirectory(curDir);
        return result;
    }

    // Moves a specified file / direcotry to a new location, 
    // providing the option to specify a new file / directory name.
    inline bool FileSystemInfo::MoveTo(const String& newPath)
    {
        if (MoveFile(_path, newPath))
        {
            _path = newPath;
            return true;
        }
        else
            return false;
    }

    ///////////////////////////////////////////////////////////////////////

    // Provides instance methods for the creation, deletion and moving of files.
    class FileInfo : public FileSystemInfo
    {
    public:
        // Initializes a new instance of the FileInfo class, 
        // which acts as a wrapper for a file path.
        FileInfo(const String& path) : FileSystemInfo(path) {};

        // Overridden. Gets a value indicating whether a file exists.
        virtual bool Exists() const;

        // Overrideen. Creates a file.
        virtual bool Create() const;

        // Overridden. Permanently deletes a file.
        virtual bool Delete() const;

        // Gets the full path of the parent directory.
        String Directory() const;
    };

    // Initializes a new instance of the FileInfo class, 
    // which acts as a wrapper for a file path.
    inline bool FileInfo::Exists() const
    {
        DWORD attributes = GetFileAttributes(_path);

        return (attributes != INVALID_FILE_ATTRIBUTES && 
                !(attributes & FILE_ATTRIBUTE_DIRECTORY));
    }

    // Overrideen. Creates a file
    inline bool FileInfo::Create() const
    {
        HANDLE handle = CreateFile(_path, GENERIC_READ, 0, NULL, CREATE_NEW, 
            FILE_ATTRIBUTE_NORMAL, NULL);

        if (handle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(handle);
            return true;
        }
        else
            return false;
    }

    // Overridden. Permanently deletes a file.
    inline bool FileInfo::Delete() const
    {
        return DeleteFile(_path) != 0;
    }

    // Gets the full path of the parent directory.
    inline String FileInfo::Directory() const
    {
        return FullName().Substring(0, _path.LastIndexOf('\\'));
    }

    ///////////////////////////////////////////////////////////////////////

    // Provides instance methods for the creation, deletion and moving of directories. 
    // Also provides instance methods for generating file and subdirectory list.
    class DirectoryInfo : public FileSystemInfo
    {
    public:
        // Initializes a new instance of the DirectoryInfo class on the specified path.
        DirectoryInfo(const String& path) : FileSystemInfo(path) {};

        // Overridden. Gets a value indicating whether the directory exists.
        virtual bool Exists() const;

        // Overridden. Creates a directory.
        virtual bool Create() const;

        // Overridden. Deletes a DirectoryInfo and its contents from a path.
        virtual bool Delete() const;

        // Gets the parent directory of a specified subdirectory.
        String Parent() const;

        // Returns the full paths of subdirectories in the current directory.
        ArrayList<String> GetDirectories() const;

        // Returns the full paths of files in the current directory.
        ArrayList<String> GetFiles() const;
    };

    // Overridden. Gets a value indicating whether the directory exists.
    inline bool DirectoryInfo::Exists() const
    {
        DWORD attributes = GetFileAttributes(_path);

        return (attributes != INVALID_FILE_ATTRIBUTES && 
                (attributes & FILE_ATTRIBUTE_DIRECTORY));
    }

    // Overridden. Creates a directory.
    inline bool DirectoryInfo::Create() const
    {
        return CreateDirectory(_path, NULL) != 0;
    }

    // Overridden. Deletes a DirectoryInfo and its contents from a path.
    inline bool DirectoryInfo::Delete() const
    {
        return RemoveDirectory(_path) != 0;
    }

    // Gets the parent directory of a specified subdirectory.
    inline String DirectoryInfo::Parent() const
    {
        return FullName().Substring(0, _path.LastIndexOf('\\'));
    }

    // Returns the full paths of subdirectories in the current directory.
    inline ArrayList<String> DirectoryInfo::GetDirectories() const
    {
        WIN32_FIND_DATA data; 
        ArrayList<String> subDirs;
        TCHAR curDir[MAX_PATH];

        GetCurrentDirectory(MAX_PATH, curDir);
        SetCurrentDirectory(_path);

        HANDLE handle = FindFirstFile("*", &data);
        if (handle != INVALID_HANDLE_VALUE)
        {
            do
            {
                if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    (data.cFileName[0] != '.')) // ignore '.' and '..'
                    subDirs.Add(_path + "\\" + data.cFileName);

            } while (FindNextFile(handle, &data));
        }

        SetCurrentDirectory(curDir);
        return subDirs;
    }

    // Returns the full paths of files in the current directory.
    inline ArrayList<String> DirectoryInfo::GetFiles() const
    {
        WIN32_FIND_DATA data; 
        ArrayList<String> files;
        TCHAR curDir[MAX_PATH];

        GetCurrentDirectory(MAX_PATH, curDir);
        SetCurrentDirectory(_path);

        HANDLE handle = FindFirstFile("*", &data);
        if (handle != INVALID_HANDLE_VALUE)
        {
            do
            {
                if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                    files.Add(_path + "\\" + data.cFileName);

            } while (FindNextFile(handle, &data));
        }

        SetCurrentDirectory(curDir);
        return files;
    }
}
}