#include "stdafx.h"
#include "CppUnitTest.h"
#include "../System/String.h"
#include "../System/FileInfo.h"
#include "../System/DirectoryInfo.h"
#include <algorithm>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace System;
using namespace System::IO;
using namespace std;

namespace UnitTest
{        
    TEST_CLASS(IOTest)
    {
    public:
        TEST_METHOD(IORelatedTest)
        {
            String rootDirFullName = "D:\\IOUnitTest";
            DirectoryInfo rootDir(rootDirFullName);

            Assert::AreEqual(false, rootDir.Exists());
            Assert::AreEqual("", rootDir.Extension());
            Assert::AreEqual("IOUnitTest", (const char*)rootDir.Name());
            Assert::AreEqual(rootDirFullName, (const char*)rootDir.FullName());
            Assert::AreEqual("D:", (const char*)rootDir.Parent());

            Assert::AreEqual(true, rootDir.Create());
            Assert::AreEqual(true, rootDir.Exists());
            Assert::AreEqual(false, rootDir.Create());

            String subDirFullName1 = rootDirFullName + "\\SubDir1";
            String subDirFullName2 = rootDirFullName + "\\SubDir2";
            DirectoryInfo subDir1(subDirFullName1);
            DirectoryInfo subDir2(subDirFullName2);

            Assert::AreEqual(false, subDir1.Exists());
            Assert::AreEqual(false, subDir2.Exists());
            Assert::AreEqual(true, subDir1.Create());
            Assert::AreEqual(true, subDir2.Create());
            Assert::AreEqual((const char*)rootDirFullName, subDir1.Parent());
            Assert::AreEqual((const char*)rootDirFullName, subDir2.Parent());

            String fileFullName1 = rootDirFullName + "\\File1.txt";
            String fileFullName2 = rootDirFullName + "\\File2.txt";
            FileInfo file1 = FileInfo(fileFullName1);
            FileInfo file2 = FileInfo(fileFullName2);
            
            Assert::AreEqual(false, file1.Exists());
            Assert::AreEqual(false, file2.Exists());
            Assert::AreEqual(true, file1.Create());
            Assert::AreEqual(true, file2.Create());
            Assert::AreEqual((const char*)rootDirFullName, file1.Directory());
            Assert::AreEqual((const char*)rootDirFullName, file2.Directory());

            vector<String> subDirs = rootDir.GetDirectories();
            sort(subDirs.begin(), subDirs.end());
            Assert::AreEqual(2, (int)subDirs.size());
            Assert::AreEqual((const char*)subDirFullName1, subDirs[0]);
            Assert::AreEqual((const char*)subDirFullName2, subDirs[1]);

            vector<String> files = rootDir.GetFiles();
            sort(files.begin(), files.end());
            Assert::AreEqual(2, (int)files.size());
            Assert::AreEqual((const char*)fileFullName1, files[0]);
            Assert::AreEqual((const char*)fileFullName2, files[1]);

            String subDirNewFullName = rootDirFullName + "\\SubDirNew";
            DirectoryInfo subDirNew(subDirNewFullName);
            String fileNewFullName = rootDirFullName + "\\FileNew.txt";
            FileInfo fileNew(fileNewFullName);

            Assert::AreEqual(true, subDir1.MoveTo(subDirNewFullName));
            Assert::AreEqual(true, file1.MoveTo(fileNewFullName));

            Assert::AreEqual(true, subDirNew.Delete());
            Assert::AreEqual(true, subDir2.Delete());
            Assert::AreEqual(true, fileNew.Delete());
            Assert::AreEqual(true, file2.Delete());
            Assert::AreEqual(true, rootDir.Delete());
        }
    };
}