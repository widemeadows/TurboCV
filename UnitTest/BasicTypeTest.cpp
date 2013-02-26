#include "stdafx.h"
#include "CppUnitTest.h"
#include "../System/Type.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace System;

namespace UnitTest
{        
    TEST_CLASS(BasicTypeTest)
    {
    public:
        TEST_METHOD(ParseTest)
        {
            Assert::AreEqual(123, Int::Parse(String("123")));
            Assert::AreEqual(123.5F, Float::Parse(String("123.5")));
            Assert::AreEqual(123.555555, Double::Parse(String("123.555555")));
        }

        TEST_METHOD(ToStringTest)
        {
            Assert::AreEqual((const char*)String("123"), Int::ToString(123));
            Assert::AreEqual((const char*)String("123.5"), Float::ToString(123.5));
            Assert::AreEqual((const char*)String("123.555555"), Double::ToString(123.555555));
        }
    };
}