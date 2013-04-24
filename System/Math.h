#pragma once

#include <cassert>
#include <cmath>
#include "Type.h"

namespace TurboCV
{
namespace System
{
    class Math
    {
    public:
        static const double PI;

        template<typename T>
        static T Min(const Vector<T>& vec);

        template<typename T>
        static T Max(const Vector<T>& vec);

        template<typename T>
        static T Sum(const Vector<T>& vec);

        template<typename T>
        static double Mean(const Vector<T>& vec);

        template<typename T>
        static double StandardDeviation(const Vector<T>& vec);

        static double Gauss(double x, double sigma);

        static double GaussDeriv(double x, double sigma);
  
        template<typename T>
        static double NormOneDistance(const Vector<T>& u, const Vector<T>& v);

        template<typename T>
        static double NormTwoDistance(const Vector<T>& u, const Vector<T>& v);

        template<typename T>
        static double GaussianDistance(const Vector<T>& u, const Vector<T>& v, double sigma);
    };

    const double Math::PI = 3.1415926535897932384626433832795;

    template<typename T>
    inline T Math::Min(const Vector<T>& vec)
    {
        assert(vec.size() > 0);

        T result = vec[0];
        for (size_t i = 1; i < vec.size(); i++)
            result = min(vec[i], result);

        return result;
    }

    template<typename T>
    inline T Math::Max(const Vector<T>& vec)
    {
        assert(vec.size() > 0);

        T result = vec[0];
        for (size_t i = 1; i < vec.size(); i++)
            result = max(vec[i], result);

        return result;
    }

    template <typename T>
    inline T Math::Sum(const Vector<T>& vec)
    {
        T sum = 0;

        for (T item : vec)
            sum += item;

        return sum;
    }

    template<typename T>
    inline double Math::Mean(const Vector<T>& vec)
    {
        assert(vec.size() > 0);

        return Sum(vec) / (double)vec.size();
    }

    template<typename T>
    inline double Math::StandardDeviation(const Vector<T>& vec)
    {
        assert(vec.size() > 0);

        double mean = Mean(vec);

        double squareSum = 0;
        for (auto item : vec)
            squareSum += (double)item * (double)item;

        return sqrt((squareSum - vec.size() * mean * mean) / vec.size());
    }

    inline double Math::Gauss(double x, double sigma)
    {
        return exp(-pow(x, 2.0) / (2 * pow(sigma, 2.0))) / (sigma * sqrt(2 * PI));
    }

    inline double Math::GaussDeriv(double x, double sigma)
    {
        return -x * Gauss(x, sigma) / pow(sigma, 2);
    }

    template<typename T>
    inline double Math::NormOneDistance(const Vector<T>& u, const Vector<T>& v)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (size_t i = 0; i < u.size(); i++)
            distance += abs(u[i] - v[i]);

        return distance;
    }

    template<typename T>
    inline double Math::NormTwoDistance(const Vector<T>& u, const Vector<T>& v)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (size_t i = 0; i < u.size(); i++)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return sqrt(distance);
    }

    template<typename T>
    inline double Math::GaussianDistance(const Vector<T>& u, const Vector<T>& v, double sigma)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (size_t i = 0; i < u.size(); i++)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return exp(-distance / (2 * sigma * sigma));
    }
}
}