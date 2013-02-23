#pragma once

#include <Windows.h>
#include "String.h"
#include "FileSystemInfo.h"

namespace System
{
    namespace IO
    {
        class DirectoryInfo;

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
    }
}