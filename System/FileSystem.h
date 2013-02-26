#pragma once

#include <Windows.h>
#include "String.h"

namespace System
{
    namespace IO
    {
        ///////////////////////////////////////////////////////////////////////

        class FileSystemInfo
        {
        public:
            FileSystemInfo(const String& path);

            String Extension() const;
            String Name() const;
            String FullName() const;

            virtual bool Exists() const = 0;
            virtual bool Create() const = 0;
            virtual bool Delete() const = 0;
            bool MoveTo(const String& newPath);

        protected:
            String _path;
        };

        inline FileSystemInfo::FileSystemInfo(const String& path)
        {
            int pathLen = path.Length();
            int lastPos = pathLen - 1;

            while (lastPos >= 0 && path[lastPos] == '\\')
                lastPos--;
            
            _path = path.Substring(0, lastPos + 1);
        }

        inline String FileSystemInfo::Extension() const
        {
            int lastBacklash = _path.LastIndexOf('\\');
            int lastDot = _path.LastIndexOf('.');

            if (lastBacklash < lastDot)
                return _path.Substring(lastDot);
            else
                return "";
        }

        inline String FileSystemInfo::Name() const
        {
            return _path.Substring(_path.LastIndexOf('\\') + 1);
        }

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
                if (fileName == NULL)
                    result = String(path);
                else
                    result = String(path) + "\\" + String(*fileName);
            }

            SetCurrentDirectory(curDir);
            return result;
        }

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

        class FileInfo : public FileSystemInfo
        {
        public:
            FileInfo(const String& path) : FileSystemInfo(path) {};

            virtual bool Exists() const;
            virtual bool Create() const;
            virtual bool Delete() const;

            String Directory() const;
        };

        inline bool FileInfo::Exists() const
        {
            DWORD attributes = GetFileAttributes(_path);

            return (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY));
        }

        inline bool FileInfo::Create() const
        {
            HANDLE handle = CreateFile(_path, GENERIC_READ, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

            if (handle != INVALID_HANDLE_VALUE)
            {
                CloseHandle(handle);
                return true;
            }
            else
                return false;
        }

        inline bool FileInfo::Delete() const
        {
            return DeleteFile(_path);
        }

        inline String FileInfo::Directory() const
        {
            return FullName().Substring(0, _path.LastIndexOf('\\'));
        }

        ///////////////////////////////////////////////////////////////////////

        class DirectoryInfo : public FileSystemInfo
        {
        public:
            DirectoryInfo(const String& path) : FileSystemInfo(path) {};

            virtual bool Exists() const;
            virtual bool Create() const;
            virtual bool Delete() const;

            String Parent() const;

            vector<String> GetDirectories() const;
            vector<String> GetFiles() const;
        };

        inline bool DirectoryInfo::Exists() const
        {
            DWORD attributes = GetFileAttributes(_path);

            return (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY));
        }

        inline bool DirectoryInfo::Create() const
        {
            return CreateDirectory(_path, NULL);
        }

        inline bool DirectoryInfo::Delete() const
        {
            return RemoveDirectory(_path);
        }

        inline String DirectoryInfo::Parent() const
        {
            return FullName().Substring(0, _path.LastIndexOf('\\'));
        }

        inline vector<String> DirectoryInfo::GetDirectories() const
        {
            WIN32_FIND_DATA data; 
            vector<String> subDirs;
            TCHAR curDir[MAX_PATH];

            GetCurrentDirectory(MAX_PATH, curDir);
            SetCurrentDirectory(_path);

            HANDLE handle = FindFirstFile("*", &data);
            if (handle != INVALID_HANDLE_VALUE)
            {
                do
                {
                    if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                        (data.cFileName[0] != '.'))
                        subDirs.push_back(_path + "\\" + data.cFileName);

                    CloseHandle(handle);
                } while (FindNextFile(handle, &data));
            }

            SetCurrentDirectory(curDir);
            return subDirs;
        }

        inline vector<String> DirectoryInfo::GetFiles() const
        {
            WIN32_FIND_DATA data; 
            vector<String> files;
            TCHAR curDir[MAX_PATH];

            GetCurrentDirectory(MAX_PATH, curDir);
            SetCurrentDirectory(_path);

            HANDLE handle = FindFirstFile("*", &data);
            if (handle != INVALID_HANDLE_VALUE)
            {
                do
                {
                    if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                        files.push_back(_path + "\\" + data.cFileName);

                    CloseHandle(handle);
                } while (FindNextFile(handle, &data));
            }

            SetCurrentDirectory(curDir);
            return files;
        }
    }
}