#include <cstdlib>
#include <iostream>
#include "String.h"
#include "DirectoryInfo.h"
using namespace std;
using namespace System;
using namespace System::IO;

int main()
{
    cout << DirectoryInfo("D:\\cygwin").FullName() << endl;

    system("pause");
}