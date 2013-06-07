#pragma once

#include <cassert>
#include <cmath>
#include "Collection.h"

namespace TurboCV
{
namespace System
{
    class Math
    {
    public:
        static const double PI;

        template<typename T>
        static T Min(const ArrayList<T>& vec);

        template<typename T>
        static T Max(const ArrayList<T>& vec);

        template<typename T>
        static T Sum(const ArrayList<T>& vec);

        template<typename T>
        static double Mean(const ArrayList<T>& vec);

        template<typename T>
        static double StandardDeviation(const ArrayList<T>& vec);

        static double Gauss(double x, double sigma);

        static double GaussDeriv(double x, double sigma);
  
        template<typename T>
        static double NormOneDistance(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static double NormTwoDistance(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static double GaussianDistance(const ArrayList<T>& u, const ArrayList<T>& v, double sigma);
    };

    template<typename T>
    inline T Math::Min(const ArrayList<T>& vec)
    {
        assert(vec.Count() > 0);

        T result = vec[0];
        for (size_t i = 1; i < vec.Count(); i++)
            result = min(vec[i], result);

        return result;
    }

    template<typename T>
    inline T Math::Max(const ArrayList<T>& vec)
    {
        assert(vec.Count() > 0);

        T result = vec[0];
        for (size_t i = 1; i < vec.Count(); i++)
            result = max(vec[i], result);

        return result;
    }

    template <typename T>
    inline T Math::Sum(const ArrayList<T>& vec)
    {
        T sum = 0;

        for (T item : vec)
            sum += item;

        return sum;
    }

    template<typename T>
    inline double Math::Mean(const ArrayList<T>& vec)
    {
        assert(vec.Count() > 0);

        return Sum(vec) / (double)vec.Count();
    }

    template<typename T>
    inline double Math::StandardDeviation(const ArrayList<T>& vec)
    {
        assert(vec.Count() > 0);

        double mean = Mean(vec);

        double squareSum = 0;
        for (auto item : vec)
            squareSum += (double)item * (double)item;

        return std::sqrt((squareSum - vec.Count() * mean * mean) / vec.Count());
    }

    inline double Math::Gauss(double x, double sigma)
    {
        return std::exp(-std::pow(x, 2.0) / (2 * std::pow(sigma, 2.0))) / (sigma * std::sqrt(2 * PI));
    }

    inline double Math::GaussDeriv(double x, double sigma)
    {
        return -x * Gauss(x, sigma) / std::pow(sigma, 2);
    }

    template<typename T>
    inline double Math::NormOneDistance(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());

        double distance = 0;
        for (size_t i = 0; i < u.Count(); i++)
            distance += std::abs(u[i] - v[i]);

        return distance;
    }

    template<typename T>
    inline double Math::NormTwoDistance(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());

        double distance = 0;
        for (size_t i = 0; i < u.Count(); i++)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return std::sqrt(distance);
    }

    template<typename T>
    inline double Math::GaussianDistance(const ArrayList<T>& u, const ArrayList<T>& v, double sigma)
    {
        assert(u.Count() == v.Count());

        double distance = 0;
        for (size_t i = 0; i < u.Count(); i++)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return std::exp(-distance / (2 * sigma * sigma));
    }
}
}