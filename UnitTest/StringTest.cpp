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
            Assert::AreEqual("", emptyString);

            const char* CString1 = "Hello World";
            String fromCString1(CString1);
            Assert::AreEqual(CString1, fromCString1);

            const char* CString2 = "";
            String fromCString2(CString2);
            Assert::AreEqual(CString2, fromCString2);

            const string StdString1 = "Hello World";
            String fromStdString1(StdString1);
            Assert::AreEqual(StdString1.c_str(), fromStdString1);

            const string StdString2 = "";
            String fromStdString2(StdString2);
            Assert::AreEqual(StdString2.c_str(), fromStdString2);

            String copyString1(fromCString1);
            Assert::AreEqual(CString1, copyString1);

            String copyString2(fromCString2);
            Assert::AreEqual("", copyString2);
        }

        TEST_METHOD(AssignmentOperatorTest)
        {
            String testString = "test";

            const char* CString1 = "Hello World";
            testString = CString1;
            Assert::AreEqual(CString1, testString);

            const char* CString2 = "";
            testString = CString2;
            Assert::AreEqual(CString2, testString);

            const string StdString1 = "Hello World";
            testString = StdString1;
            Assert::AreEqual(StdString1.c_str(), testString);

            const string StdString2 = "";
            testString = StdString2;
            Assert::AreEqual(StdString2.c_str(), testString);

            const String copyString1(CString1);
            testString = copyString1;
            Assert::AreEqual(CString1, testString);

            const String copyString2(CString2);
            testString = copyString2;
            Assert::AreEqual(CString2, testString);

            testString = "Self Copy";
            testString = testString;
            Assert::AreEqual("Self Copy", testString);
        }

        TEST_METHOD(AddOperatorTest)
        {
            String emptyString;
            String nonEmptyString = "Hello";

            Assert::AreEqual("", emptyString + "");
            Assert::AreEqual("", "" + emptyString);

            Assert::AreEqual("", emptyString + string());
            Assert::AreEqual("", string() + emptyString);

            Assert::AreEqual("", emptyString + emptyString);

            Assert::AreEqual("Hello", emptyString + "Hello");
            Assert::AreEqual("Hello", "Hello" + emptyString);

            Assert::AreEqual("Hello", emptyString + string("Hello"));
            Assert::AreEqual("Hello", string("Hello") + emptyString);

            Assert::AreEqual("Hello", emptyString + nonEmptyString);
            Assert::AreEqual("Hello", nonEmptyString + emptyString);

            Assert::AreEqual("Hello World", nonEmptyString + " World");
            Assert::AreEqual("World Hello", "World " + nonEmptyString);

            Assert::AreEqual("Hello World", nonEmptyString + string(" World"));
            Assert::AreEqual("World Hello", string("World ") + nonEmptyString);

            Assert::AreEqual("Hello World", nonEmptyString + String(" World"));
            Assert::AreEqual("World Hello", String("World ") + nonEmptyString);
        }

        TEST_METHOD(RelationOperatorTest)
        {
            Assert::AreEqual(true, String() < String("abc"));
            Assert::AreEqual(true, String("ab") < String("abc"));
            Assert::AreEqual(true, String("abb") < String("abc"));
            Assert::AreEqual(false, String("abd") < String("abc"));
            Assert::AreEqual(false, String("abc") < String("abc"));
            Assert::AreEqual(false, String("ad") < String("abc"));

            Assert::AreEqual(true, String() == String());
            Assert::AreEqual(true, String("abc") == String("abc"));
            Assert::AreEqual(false, String() == String("abc"));
            Assert::AreEqual(false, String("abd") == String("abc"));
        }

        TEST_METHOD(ConversionOperatorTest)
        {
            Assert::AreEqual("", (const char*)String(""));
            Assert::AreEqual("abc", (const char*)String("abc"));

            Assert::AreEqual(true, string("") == String("").operator std::string());
            Assert::AreEqual(true , string("abc") == String("abc").operator std::string());
        }

        TEST_METHOD(SubstringTest)
        {
            String str = "Hello World";

            Assert::AreEqual("Hello World", str.Substring(0));
            Assert::AreEqual("World", str.Substring(6));
            Assert::AreEqual("", str.Substring(11));

            Assert::AreEqual("Hello", str.Substring(0, 5));
            Assert::AreEqual(" ", str.Substring(5, 1));
            Assert::AreEqual("", str.Substring(5, 0));

            str = "";

            Assert::AreEqual("", str.Substring(0));
            Assert::AreEqual("", str.Substring(0, 0));
        }

        TEST_METHOD(IndexOfTest)
        {
            String str = "Hello World; Hi World.";

            Assert::AreEqual(0, str.IndexOf('H'));
            Assert::AreEqual(14, str.IndexOf('i'));
            Assert::AreEqual(21, str.IndexOf('.'));
            Assert::AreEqual(-1, str.IndexOf('!'));

            Assert::AreEqual(0, str.IndexOf("Hello"));
            Assert::AreEqual(6, str.IndexOf("World"));
            Assert::AreEqual(21, str.IndexOf("."));
            Assert::AreEqual(-1, str.IndexOf("NotExist"));
        }

        TEST_METHOD(SplitTest)
        {
            String expected[] = { "Good", "Morning", "Hello", "World" };
            vector<String> actual;
            String test;
            
            test = "Good Morning, Hello World!";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.size());
            for (int i = 0; i < actual.size(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good   Morning,  Hello  World!  ";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.size());
            for (int i = 0; i < actual.size(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good   Morning   Hello  World   ";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.size());
            for (int i = 0; i < actual.size(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good Morning  Hello World     ";
            actual = test.Split(' ');
            Assert::AreEqual(4, (int)actual.size());
            for (int i = 0; i < actual.size(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "              ";
            actual = test.Split(" ,!");
            Assert::AreEqual(0, (int)actual.size());

            test = "              ";
            actual = test.Split(' ');
            Assert::AreEqual(0, (int)actual.size());

            test = "";
            actual = test.Split(" ,!");
            Assert::AreEqual(0, (int)actual.size());

            test = "";
            actual = test.Split(' ');
            Assert::AreEqual(0, (int)actual.size());

            test = "Hello World!";
            actual = test.Split(",");
            Assert::AreEqual(1, (int)actual.size());
            Assert::AreEqual((const char*)test, (const char*)actual[0]);

            test = "Hello World!";
            actual = test.Split(',');
            Assert::AreEqual(1, (int)actual.size());
            Assert::AreEqual((const char*)test, (const char*)actual[0]);
        }
    };
}