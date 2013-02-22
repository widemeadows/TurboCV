#include "stdafx.h"
#include "CppUnitTest.h"
#include "../System/String.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace System;

namespace UnitTest
{		
	TEST_CLASS(StringTest)
	{
	public:
		TEST_METHOD(ConstructorTest)
		{
			String emptyString;
			Assert::AreEqual(emptyString.Length(), 0);

			const char* cstring = "Hello World";
			String fromCString(cstring);
			Assert::AreEqual(cstring, (char*)fromCString);

			const string stdstring = "Hello World";
			String fromStdString(stdstring);
			Assert::AreEqual(stdstring, (string)fromStdString);
		}

	};
}