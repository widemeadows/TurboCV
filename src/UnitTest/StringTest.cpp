#include "stdafx.h"
#include "CppUnitTest.h"
#include "../System/Collection.h"
#include "../System/Type.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace TurboCV::System;

namespace UnitTest
{        
    TEST_CLASS(StringTest)
    {
    public:
        TEST_METHOD(ConstructorTest)
        {
            TString emptyString;
            Assert::AreEqual("", emptyString);

            const char* CString1 = "Hello World";
            TString fromCString1(CString1);
            Assert::AreEqual(CString1, fromCString1);

            const char* CString2 = "";
            TString fromCString2(CString2);
            Assert::AreEqual(CString2, fromCString2);

            const string StdString1 = "Hello World";
            TString fromStdString1(StdString1);
            Assert::AreEqual(StdString1.c_str(), fromStdString1);

            const string StdString2 = "";
            TString fromStdString2(StdString2);
            Assert::AreEqual(StdString2.c_str(), fromStdString2);

            TString copyString1(fromCString1);
            Assert::AreEqual(CString1, copyString1);

            TString copyString2(fromCString2);
            Assert::AreEqual("", copyString2);
        }

        TEST_METHOD(AssignmentOperatorTest)
        {
            TString testString = "test";

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

            const TString copyString1(CString1);
            testString = copyString1;
            Assert::AreEqual(CString1, testString);

            const TString copyString2(CString2);
            testString = copyString2;
            Assert::AreEqual(CString2, testString);

            testString = "Self Copy";
            testString = testString;
            Assert::AreEqual("Self Copy", testString);
        }

        TEST_METHOD(AddOperatorTest)
        {
            TString emptyString;
            TString nonEmptyString = "Hello";

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

            Assert::AreEqual("Hello World", nonEmptyString + TString(" World"));
            Assert::AreEqual("World Hello", TString("World ") + nonEmptyString);
        }

        TEST_METHOD(RelationOperatorTest)
        {
            Assert::AreEqual(true, TString() < TString("abc"));
            Assert::AreEqual(true, TString("ab") < TString("abc"));
            Assert::AreEqual(true, TString("abb") < TString("abc"));
            Assert::AreEqual(false, TString("abd") < TString("abc"));
            Assert::AreEqual(false, TString("abc") < TString("abc"));
            Assert::AreEqual(false, TString("ad") < TString("abc"));

            Assert::AreEqual(true, TString() == TString());
            Assert::AreEqual(true, TString("abc") == TString("abc"));
            Assert::AreEqual(false, TString() == TString("abc"));
            Assert::AreEqual(false, TString("abd") == TString("abc"));
        }

        TEST_METHOD(ConversionOperatorTest)
        {
            Assert::AreEqual("", (const char*)TString(""));
            Assert::AreEqual("abc", (const char*)TString("abc"));

            Assert::AreEqual(true, string("") == TString("").operator std::string());
            Assert::AreEqual(true , string("abc") == TString("abc").operator std::string());
        }

        TEST_METHOD(SubstringTest)
        {
            TString str = "Hello World";

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

        TEST_METHOD(StrSearchTest)
        {
            TString str = "Hello World; Hi World.";

            Assert::AreEqual(0ULL, str.IndexOf('H'));
            Assert::AreEqual(14ULL, str.IndexOf('i'));
            Assert::AreEqual(21ULL, str.IndexOf('.'));
            Assert::AreEqual((size_t)-1, str.IndexOf('!'));

            Assert::AreEqual(0ULL, str.IndexOf("Hello"));
            Assert::AreEqual(6ULL, str.IndexOf("World"));
            Assert::AreEqual(21ULL, str.IndexOf("."));
            Assert::AreEqual((size_t)-1, str.IndexOf("NotExist"));

            str = "!Oh, Good morning. Good";

            Assert::AreEqual(19ULL, str.LastIndexOf('G'));
            Assert::AreEqual(3ULL, str.LastIndexOf(','));
            Assert::AreEqual(0ULL, str.LastIndexOf('!'));
            Assert::AreEqual((size_t)-1, str.LastIndexOf('?'));

            Assert::AreEqual(19ULL, str.LastIndexOf("Good"));
            Assert::AreEqual(10ULL, str.LastIndexOf("morning"));
            Assert::AreEqual(0ULL, str.LastIndexOf("!"));
            Assert::AreEqual((size_t)-1, str.LastIndexOf("NotExist"));
        }

        TEST_METHOD(SplitTest)
        {
            TString expected[] = { "Good", "Morning", "Hello", "World" };
            ArrayList<TString> actual;
            TString test;
            
            test = "Good Morning, Hello World!";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.Count());
            for (int i = 0; i < actual.Count(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good   Morning,  Hello  World!  ";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.Count());
            for (int i = 0; i < actual.Count(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good   Morning   Hello  World   ";
            actual = test.Split(" ,!");
            Assert::AreEqual(4, (int)actual.Count());
            for (int i = 0; i < actual.Count(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "  Good Morning  Hello World     ";
            actual = test.Split(' ');
            Assert::AreEqual(4, (int)actual.Count());
            for (int i = 0; i < actual.Count(); i++)
                Assert::AreEqual((const char*)expected[i], (const char*)actual[i]);

            test = "              ";
            actual = test.Split(" ,!");
            Assert::AreEqual(0, (int)actual.Count());

            test = "              ";
            actual = test.Split(' ');
            Assert::AreEqual(0, (int)actual.Count());

            test = "";
            actual = test.Split(" ,!");
            Assert::AreEqual(0, (int)actual.Count());

            test = "";
            actual = test.Split(' ');
            Assert::AreEqual(0, (int)actual.Count());

            test = "Hello World!";
            actual = test.Split(",");
            Assert::AreEqual(1, (int)actual.Count());
            Assert::AreEqual((const char*)test, (const char*)actual[0]);

            test = "Hello World!";
            actual = test.Split(',');
            Assert::AreEqual(1, (int)actual.Count());
            Assert::AreEqual((const char*)test, (const char*)actual[0]);
        }
    };
}