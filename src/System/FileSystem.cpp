#include "Core.h"
#include <windows.h>

namespace TurboCV
{
namespace System
{
    //////////////////////////////////////////////////////////////////////////
    // FileSystemInfo
    //////////////////////////////////////////////////////////////////////////

    // Initializes a new instance of the FileSystemInfo class.
    FileSystemInfo::FileSystemInfo(const TString& path)
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
    TString FileSystemInfo::Extension() const
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
    TString FileSystemInfo::Name() const
    {
        return _path.Substring(_path.LastIndexOf('\\') + 1);
    }

    // Gets the full path of the directory or file.
    TString FileSystemInfo::FullName() const
    {
        TCHAR path[MAX_PATH] = TEXT(""); 
        TCHAR** fileName = { NULL };
        TCHAR curDir[MAX_PATH];
        TString result;

        GetCurrentDirectory(MAX_PATH, curDir);
        SetCurrentDirectory(_path);

        if (GetFullPathName(_path, MAX_PATH, path, fileName))
        {
            if (fileName == NULL) // if it is a directory
                result = TString(path);
            else // otherwise it is a file
                result = TString(path) + "\\" + TString(*fileName);
        }

        SetCurrentDirectory(curDir);
        return result;
    }

    // Moves a specified file / direcotry to a new location, 
    // providing the option to specify a new file / directory name.
    bool FileSystemInfo::MoveTo(const TString& newPath)
    {
        if (MoveFile(_path, newPath))
        {
            _path = newPath;
            return true;
        }
        else
            return false;
    }


    //////////////////////////////////////////////////////////////////////////
    // FileInfo
    //////////////////////////////////////////////////////////////////////////

    // Initializes a new instance of the FileInfo class, 
    // which acts as a wrapper for a file path.
    bool FileInfo::Exists() const
    {
        DWORD attributes = GetFileAttributes(_path);

        return (attributes != INVALID_FILE_ATTRIBUTES && 
            !(attributes & FILE_ATTRIBUTE_DIRECTORY));
    }

    // Overrideen. Creates a file
    bool FileInfo::Create() const
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
    bool FileInfo::Delete() const
    {
        return DeleteFile(_path) != 0;
    }

    // Gets the full path of the parent directory.
    TString FileInfo::Directory() const
    {
        return FullName().Substring(0, _path.LastIndexOf('\\'));
    }


    //////////////////////////////////////////////////////////////////////////
    // DirectoryInfo
    //////////////////////////////////////////////////////////////////////////

    // Overridden. Gets a value indicating whether the directory exists.
    bool DirectoryInfo::Exists() const
    {
        DWORD attributes = GetFileAttributes(_path);

        return (attributes != INVALID_FILE_ATTRIBUTES && 
            (attributes & FILE_ATTRIBUTE_DIRECTORY));
    }

    // Overridden. Creates a directory.
    bool DirectoryInfo::Create() const
    {
        return CreateDirectory(_path, NULL) != 0;
    }

    // Overridden. Deletes a DirectoryInfo and its contents from a path.
    bool DirectoryInfo::Delete() const
    {
        return RemoveDirectory(_path) != 0;
    }

    // Gets the parent directory of a specified subdirectory.
    TString DirectoryInfo::Parent() const
    {
        return FullName().Substring(0, _path.LastIndexOf('\\'));
    }

    // Returns the full paths of subdirectories in the current directory.
    ArrayList<TString> DirectoryInfo::GetDirectories() const
    {
        WIN32_FIND_DATA data; 
        ArrayList<TString> subDirs;
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
    ArrayList<TString> DirectoryInfo::GetFiles() const
    {
        WIN32_FIND_DATA data; 
        ArrayList<TString> files;
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