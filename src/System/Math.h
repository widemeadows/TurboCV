#pragma once

#include <cassert>
#include <cmath>
#include "Core.h"

namespace TurboCV
{
namespace System
{
    class Math
    {
    public:
        static const double PI;

        static double Gauss(double x, double sigma);

        static double GaussDeriv(double x, double sigma);

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

        template<typename T>
        static double NormOne(const ArrayList<T>& vec);

        template<typename T>
        static double NormTwo(const ArrayList<T>& vec);

        template<typename T>
        static ArrayList<T> Add(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static ArrayList<T> Sub(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static T Dot(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static ArrayList<T> Mul(const ArrayList<T>& u, double scalar);

        template<typename T>
        static ArrayList<T> Mul(double scalar, const ArrayList<T>& u);
  
        template<typename T>
        static double NormOneDistance(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static double NormTwoDistanceSqr(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static double NormTwoDistance(const ArrayList<T>& u, const ArrayList<T>& v);

        template<typename T>
        static double GaussianDistance(const ArrayList<T>& u, const ArrayList<T>& v, double sigma);
    };

    inline double Math::Gauss(double x, double sigma)
    {
        return std::exp(-std::pow(x, 2.0) / (2 * std::pow(sigma, 2.0))) / (sigma * std::sqrt(2 * PI));
    }

    inline double Math::GaussDeriv(double x, double sigma)
    {
        return -x * Gauss(x, sigma) / std::pow(sigma, 2);
    }

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

    template<typename T>
    inline double Math::NormOne(const ArrayList<T>& vec)
    {
        double result = 0;

        for (auto item : vec)
            result += std::abs(item);

        return result;
    }

    template<typename T>
    inline double Math::NormTwo(const ArrayList<T>& vec)
    {
        double result = 0;

        for (auto item : vec)
            result += item * item;

        if (result)
            result = std::sqrt(result);

        return result;
    }

    template<typename T>
    inline ArrayList<T> Math::Add(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());
        ArrayList<T> result(u.Count());

        for (int i = u.Count() - 1; i >= 0; i--)
            result[i] = u[i] + v[i];

        return result;
    }

    template<typename T>
    inline ArrayList<T> Math::Sub(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());
        ArrayList<T> result(u.Count());

        for (int i = u.Count() - 1; i >= 0; i--)
            result[i] = u[i] - v[i];

        return result;
    }

    template<typename T>
    inline T Math::Dot(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());
        T result = 0;

        for (int i = u.Count() - 1; i >= 0; i--)
            result += u[i] * v[i];

        return result;
    }

    template<typename T>
    inline ArrayList<T> Math::Mul(const ArrayList<T>& u, double scalar)
    {
        ArrayList<T> result(u.Count());

        for (int i = u.Count() - 1; i >= 0; i--)
            result[i] = u[i] * scalar;

        return result;
    }

    template<typename T>
    inline ArrayList<T> Math::Mul(double scalar, const ArrayList<T>& u)
    {
        return Mul(u, scalar);
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
    inline double Math::NormTwoDistanceSqr(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        assert(u.Count() == v.Count());

        double distanceSqr = 0;
        for (size_t i = 0; i < u.Count(); i++)
            distanceSqr += (u[i] - v[i]) * (u[i] - v[i]);

        return distanceSqr;
    }

    template<typename T>
    inline double Math::NormTwoDistance(const ArrayList<T>& u, const ArrayList<T>& v)
    {
        return std::sqrt(Math::NormTwoDistanceSqr(u, v));
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