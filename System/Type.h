#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
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

    ///////////////////////////////////////////////////////////////////////

    class Float
    {
    public:
        static float Parse(const String& floating);
        static String ToString(float floating);
    };

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

    ///////////////////////////////////////////////////////////////////////

    class Double
    {
    public:
        static double Parse(const String& floating);
        static String ToString(double floating);
    };

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

    ///////////////////////////////////////////////////////////////////////

    template<typename T>
    class Vector
    {
    public:
        Vector() : ptr(new vector<T>()) {};
        Vector(size_t size) : ptr(new vector<T>(size)) {};

        T& operator[](int index) { return (*ptr)[index]; };
        const T& operator[](int index) const { return (*ptr)[index]; };

        const vector<T>& getVec() const { return *ptr; }

        typename vector<T>::iterator begin() const { return ptr->begin(); };
        typename vector<T>::iterator end() const { return ptr->end(); };

        void push_back(const T& item) { ptr->push_back(item); };
        void clear() { ptr->clear(); };

        size_t size() const { return ptr->size(); };

    private:
        shared_ptr<vector<T>> ptr;
    };
}