#include "stdafx.h"
#include "CppUnitTest.h"
#include "../System/Collection.h"
#include "../System/Math.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace TurboCV::System;

namespace UnitTest
{        
    TEST_CLASS(MathTest)
    {
    public:
        TEST_METHOD(SumTest)
        {
            ArrayList<int> vec(1);
            Assert::AreEqual(0, Math::Sum(vec));

            for (int i = 1; i < 10; i++)
                vec.Add(0);
            Assert::AreEqual(0, Math::Sum(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i;
            Assert::AreEqual(45, Math::Sum(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i % 2 == 0 ? -1 : 1;
            Assert::AreEqual(0, Math::Sum(vec));
        }

        TEST_METHOD(MeanTest)
        {
            ArrayList<int> vec(1);
            Assert::AreEqual(0.0, Math::Mean(vec));

            for (int i = 1; i < 10; i++)
                vec.Add(0);
            Assert::AreEqual(0.0, Math::Mean(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i;
            Assert::AreEqual(4.5, Math::Mean(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i % 2 == 0 ? -1 : 1;
            Assert::AreEqual(0.0, Math::Mean(vec));
        }

        TEST_METHOD(StandardDeviationTest)
        {
            ArrayList<int> vec(1);
            Assert::AreEqual(0.0, Math::StandardDeviation(vec));

            for (int i = 1; i < 10; i++)
                vec.Add(0);
            Assert::AreEqual(0.0, Math::StandardDeviation(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i;
            Assert::AreEqual(sqrt(8.25), Math::StandardDeviation(vec));

            for (int i = 0; i < 10; i++)
                vec[i] = i % 2 == 0 ? -1 : 1;
            Assert::AreEqual(1.0, Math::StandardDeviation(vec));
        }

        TEST_METHOD(DistanceTest)
        {
            ArrayList<int> u, v;
            for (int i = 0; i < 10; i++)
            {
                u.Add(1);
                v.Add(i + 1);
            }

            Assert::AreEqual(45.0, Math::NormOneDistance(u, v));
            Assert::AreEqual(sqrt(285), Math::NormTwoDistance(u, v));
            Assert::AreEqual(exp(-285 / 200.0), Math::GaussianDistance(u, v, 10));
        }
    };
}