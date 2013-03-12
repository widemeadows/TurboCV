#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
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
    class ThreadUnsafeCounter
    {
    public:
        explicit ThreadUnsafeCounter(T* object) : count(1), instance(object) {};

        void AddRef() { count++; }

        void Release() 
        { 
            count--; 
            if (!count)
            {
                Dispose();
                delete this;
            }
        }

    protected:
        void Dispose() { delete instance; }

    private:
        int count;
        T* instance;
    };

    template<typename T>
    class ThreadUnsafeSmartPtr
    {
    public:
        explicit ThreadUnsafeSmartPtr() : instance(NULL), counter(NULL) {};

        explicit ThreadUnsafeSmartPtr(T* object) : 
            instance(object), counter(new ThreadUnsafeCounter<T>(object)) {};

        explicit ThreadUnsafeSmartPtr(const ThreadUnsafeSmartPtr& u) : 
            instance(u.instance), counter(u.counter)
        {
            if (counter)
                counter->AddRef();
        }

        ~ThreadUnsafeSmartPtr()
        {
            if (counter)
                counter->Release();
        }

        ThreadUnsafeSmartPtr& operator=(const ThreadUnsafeSmartPtr& u)
        {
            instance = u.instance;

            if (counter != u.counter)
            {
                if (counter)
                    counter->Release();
                counter = u.counter;
            }

            if (counter)
                counter->AddRef();

            return *this;
        }

        T* operator->() const { return instance; }
        T& operator*() const { return *instance; }

    private:
        T* instance;
        ThreadUnsafeCounter<T>* counter;
    };

    ///////////////////////////////////////////////////////////////////////

    template<typename T>
    class Vector
    {
    public:
        Vector() : ptr(new vector<T>()) {};
        Vector(size_t size) : ptr(new vector<T>(size)) {};

        T& operator[](int index) { return (*ptr)[index]; };
        const T& operator[](int index) const { return (*ptr)[index]; };

        operator const vector<T>&() const { return *ptr; }

        typename vector<T>::iterator begin() const { return ptr->begin(); };
        typename vector<T>::iterator end() const { return ptr->end(); };

        void push_back(const T& item) { ptr->push_back(item); };
        void clear() { ptr->clear(); };

        size_t size() const { return ptr->size(); };

    private:
        ThreadUnsafeSmartPtr<vector<T>> ptr;
    };
}