#pragma once

#include <vector>

namespace TurboCV
{
namespace System
{
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
    class ArrayList
    {
    public:
        ArrayList() : ptr(new std::vector<T>()) {}
        ArrayList(size_t size) : ptr(new std::vector<T>(size)) {}

        template<typename RandomAccessIterator>
        ArrayList(RandomAccessIterator begin, RandomAccessIterator end) :
            ptr(new std::vector<T>(begin, end)) {}

        T& operator[](int index) { return (*ptr)[index]; }
        const T& operator[](int index) const { return (*ptr)[index]; }

        operator const std::vector<T>&() const { return *ptr; }

        template<typename U>
        ArrayList<T> operator*(const U& factor) const
        {
            size_t length = ptr->size();
            ArrayList<T> result(length);

            for (size_t i = 0; i < length; i++)
                result[i] = (*ptr)[i] * factor;

            return result;
        }

        T& front() { return ptr->front(); }
        const T& front() const { return ptr->front(); }

        T& back() { return ptr->back(); }
        const T& back() const { return ptr->back(); }

        typename std::vector<T>::iterator begin() const { return ptr->begin(); }
        typename std::vector<T>::iterator end() const { return ptr->end(); }

        void push_back(const T& item) { ptr->push_back(item); }

        void push_back(const ArrayList<T>& vec) 
        { 
            ptr->insert(ptr->end(), vec.begin(), vec.end()); 
        }

        void push_back(
            typename const std::vector<T>::iterator& begin, 
            typename const std::vector<T>::iterator& end) 
        { 
            ptr->insert(ptr->end(), begin, end); 
        }

        void clear() { ptr->clear(); }

        size_t size() const { return ptr->size(); }

    private:
        ThreadUnsafeSmartPtr<std::vector<T>> ptr;
    };
}
}