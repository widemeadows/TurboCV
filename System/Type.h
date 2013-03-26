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

    class NullType {};

    template<typename T1 = NullType, typename T2 = NullType, typename T3 = NullType, typename T4 = NullType>
    class Tuple
    {

    };

    template<typename T1, typename T2>
    class Tuple<T1, T2, NullType, NullType>
    {
    public:
        Tuple() {}
        Tuple(const T1& i1, const T2& i2) : item1(i1), item2(i2) {}

        T1& Item1() { return item1; }
        T2& Item2() { return item2; }
        const T1& Item1() const { return item1; }
        const T2& Item2() const { return item2; }

        bool operator<(const Tuple<T1, T2>& v) const
        {
            if (item1 < v.item1)
                return true;
            else if (!(v.item1 < item1)) // item1 == v.item1
                return item2 < v.item2;
            else
                return false;
        }

    private:
        T1 item1;
        T2 item2;
    };

    template<typename T1, typename T2>
    Tuple<T1, T2> CreateTuple(const T1& i1, const T2& i2) 
    { 
        return Tuple<T1, T2>(i1, i2); 
    }

    template<typename T1, typename T2, typename T3>
    class Tuple<T1, T2, T3, NullType>
    {
    public:
        Tuple() {}
        Tuple(const T1& i1, const T2& i2, const T3& i3) : item1(i1), item2(i2), item3(i3) {}

        T1& Item1() { return item1; }
        T2& Item2() { return item2; }
        T3& Item3() { return item3; }
        const T1& Item1() const { return item1; }
        const T2& Item2() const { return item2; }
        const T3& Item3() const { return item3; }

        bool operator<(const Tuple<T1, T2, T3>& v) const
        {
            if (item1 < v.item1)
                return true;
            else if (!(v.item1 < item1)) // item1 == v.item1
            {
                if (item2 < v.item2)
                    return true;
                else if (!(v.item2 < item2)) // item2 == v.item2
                    return item3 < v.Item3;
                else
                    return false;
            }
            else
                return false;
        }

    private:
        T1 item1;
        T2 item2;
        T3 item3;
    };

    template<typename T1, typename T2, typename T3>
    Tuple<T1, T2, T3> CreateTuple(const T1& i1, const T2& i2, const T3& i3) 
    { 
        return Tuple<T1, T2, T3>(i1, i2, i3); 
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
        Vector() : ptr(new vector<T>()) {}
        Vector(size_t size) : ptr(new vector<T>(size)) {}

        T& operator[](int index) { return (*ptr)[index]; }
        const T& operator[](int index) const { return (*ptr)[index]; }

        operator const vector<T>&() const { return *ptr; }

        typename vector<T>::iterator begin() const { return ptr->begin(); }
        typename vector<T>::iterator end() const { return ptr->end(); }

        void push_back(const T& item) { ptr->push_back(item); }
        void push_back(typename const vector<T>::iterator& begin,
            typename const vector<T>::iterator& end) { ptr->insert(ptr->end(), begin, end); }
        void clear() { ptr->clear(); }

        size_t size() const { return ptr->size(); }

    private:
        ThreadUnsafeSmartPtr<vector<T>> ptr;
    };
}