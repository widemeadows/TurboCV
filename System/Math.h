#pragma once

#include <cassert>
#include <cmath>
using namespace std;

namespace System
{
    class Math
    {
    public:
        static const double PI;

        template<typename T>
        static T Sum(const vector<T>& vec);

        static double Gauss(double x, double sigma);

        static double GaussDeriv(double x, double sigma);
  
        template<typename T>
        static double NormOneDistance(const vector<T>& u, const vector<T>& v);

        template<typename T>
        static double NormTwoDistance(const vector<T>& u, const vector<T>& v);

        template<typename T>
        static double GaussianDistance(const vector<T>& u, const vector<T>& v, double sigma);
    };

    const double Math::PI = 3.1415926535897932384626433832795;

    template <typename T>
    inline T Math::Sum(const vector<T>& vec)
    {
        T sum = 0;

        for (int i = vec.size() - 1; i >= 0; i--)
            sum += vec[i];

        return sum;
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
    inline double Math::NormOneDistance(const vector<T>& u, const vector<T>& v)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (int i = u.size() - 1; i >= 0; i--)
            distance += abs(u[i] - v[i]);

        return distance;
    }

    template<typename T>
    inline double Math::NormTwoDistance(const vector<T>& u, const vector<T>& v)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (int i = u.size() - 1; i >= 0; i--)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return sqrt(distance);
    }

    template<typename T>
    inline double Math::GaussianDistance(const vector<T>& u, const vector<T>& v, double sigma)
    {
        assert(u.size() == v.size());

        double distance = 0;
        for (int i = u.size() - 1; i >= 0; i--)
            distance += (u[i] - v[i]) * (u[i] - v[i]);

        return exp(-distance / (2 * sigma * sigma));
    }
}