#pragma once

#include <vector>
#include <windows.h>

namespace TurboCV
{
namespace System
{
    template<typename T>
    class ThreadSafeCounter
    {
    public:
        explicit ThreadSafeCounter(T* object) : count(1), instance(object) {};

        void AddRef() 
        { 
            InterlockedIncrement(&count); 
        }

        void Release() 
        { 
            InterlockedDecrement(&count);

            if (!count)
            {
                Dispose();
                delete this;
            }
        }

    protected:
        void Dispose() { delete instance; }

    private:
        unsigned int count;
        T* instance;
    };

    template<typename T>
    class ThreadSafeSmartPtr
    {
    public:
        explicit ThreadSafeSmartPtr() : instance(NULL), counter(NULL) {};

        explicit ThreadSafeSmartPtr(T* object) : 
            instance(object), counter(new ThreadSafeCounter<T>(object)) {};

        explicit ThreadSafeSmartPtr(const ThreadSafeSmartPtr& u) : 
            instance(u.instance), counter(u.counter)
        {
            if (counter)
                counter->AddRef();
        }

        ~ThreadSafeSmartPtr()
        {
            if (counter)
                counter->Release();
        }

        ThreadSafeSmartPtr& operator=(const ThreadSafeSmartPtr& u)
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
        ThreadSafeCounter<T>* counter;
    };

    ///////////////////////////////////////////////////////////////////////

    template<typename T>
    class ArrayList
    {
    public:
        ArrayList() : ptr(new std::vector<T>()) {}

        ArrayList(size_t size) : ptr(new std::vector<T>(size)) {}

		ArrayList(size_t size, const T& val) : ptr(new std::vector<T>(size, val)) {}

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

        T& Front() { return ptr->front(); }
        const T& Front() const { return ptr->front(); }

        T& Back() { return ptr->back(); }
        const T& Back() const { return ptr->back(); }

        typename std::vector<T>::iterator begin() const { return ptr->begin(); }
        typename std::vector<T>::iterator end() const { return ptr->end(); }

        void Add(const T& item) { ptr->push_back(item); }

        void Add(const ArrayList<T>& vec) 
        { 
            ptr->insert(ptr->end(), vec.begin(), vec.end()); 
        }

        template<typename RandomAccessIterator>
        void Add(RandomAccessIterator begin, RandomAccessIterator end) 
        { 
            ptr->insert(ptr->end(), begin, end); 
        }

        bool Contains(const T& item) const
        {
            typename std::vector<T>::iterator begin = ptr->begin();
            typename std::vector<T>::iterator end = ptr->end();
            typename std::vector<T>::iterator itr = begin;

            while (itr != end)
            {
                if (*itr == item)
                    return true;
                itr++;
            }

            return false;
        }

        void Shrink() { ptr->shrink_to_fit(); }

        void Clear() { ptr->clear(); }

        size_t Count() const { return ptr->size(); }

    private:
        ThreadSafeSmartPtr<std::vector<T>> ptr;
    };
}
}