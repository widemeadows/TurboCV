#pragma once

#include <Windows.h>
#include "String.h"
#include "FileSystemInfo.h"

namespace System
{
    namespace IO
    {
        class FileInfo;

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