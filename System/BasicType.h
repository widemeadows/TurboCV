#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include "String.h"
using namespace std;

namespace System
{
    class Int
    {
    public:
        static int Parse(const String& integer);
        static String ToString(int integer);
    };

    class Float
    {
    public:
        static float Parse(const String& floating);
        static String ToString(float floating);
    };

    class Double
    {
    public:
        static double Parse(const String& floating);
        static String ToString(double floating);
    };

    inline int Int::Parse(const String& integer)
    {
        stringstream ss;
        int result;

        ss << integer;
        ss >> result;

        return result;
    }

    inline String Int::ToString(int integer)
    {
        stringstream ss;
        String result;

        ss << integer;
        ss >> result;

        return result;
    }

    inline float Float::Parse(const String& floating)
    {
        stringstream ss;
        float result;

        ss << setprecision(7) << floating;
        ss >> result;

        return result;
    }

    inline String Float::ToString(float floating)
    {
        stringstream ss;
        String result;

        ss << setprecision(7) << floating;
        ss >> result;

        return result;
    }

    inline double Double::Parse(const String& floating)
    {
        stringstream ss;
        double result;

        ss << setprecision(16) << floating;
        ss >> result;

        return result;
    }

    inline String Double::ToString(double floating)
    {
        stringstream ss;
        String result;

        ss << setprecision(16) << floating;
        ss >> result;

        return result;
    }
}