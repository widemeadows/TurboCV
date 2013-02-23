#pragma once

#include <Windows.h>
#include "String.h"

namespace System
{
    namespace IO
    {
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
    }
}