#pragma once

#include <memory>
#include <vector>
#include <iterator>
using namespace std;

namespace System
{
    namespace Image
    {
        template<typename T>
        class DescriptorInfo
        {
        public:
            DescriptorInfo() : ptr(new vector<T>()) {};

            typename T& operator[](int index) { return (*ptr)[index]; };
            typename const T& operator[](int index) const { return (*ptr)[index]; };

            typename vector<T> getVec() const { return *ptr; }

            typename vector<T>::iterator begin() const { return ptr->begin(); };
            typename vector<T>::iterator end() const { return ptr->end(); };

            void push_back(const T& item) { ptr->push_back(item); };
            void clear() { ptr->clear(); };

            int size() const { return (int)(ptr->size()); };

        private:
            shared_ptr<vector<T>> ptr;
        };

        template<typename T>
        class FeatureInfo
        {
        public:
            FeatureInfo() : ptr(new vector<DescriptorInfo<T>>()) {};

            typename DescriptorInfo<T>& operator[](int index) { return (*ptr)[index]; };
            typename const DescriptorInfo<T>& operator[](int index) const { return (*ptr)[index]; };

            typename vector<DescriptorInfo<T>>::iterator begin() const { return ptr->begin(); };
            typename vector<DescriptorInfo<T>>::iterator end() const { return ptr->end(); };

            void push_back(const DescriptorInfo<T>& item) { ptr->push_back(item); };
            void clear() { ptr->clear(); };

            int size() const { return (int)(ptr->size()); };

        private:
            shared_ptr<vector<DescriptorInfo<T>>> ptr;
        };

        template<typename T1, typename T2>
        inline void Convert(const FeatureInfo<T1>& src, FeatureInfo<T2>& dst)
        {
            dst.clear();

            for (auto descriptor : src)
            {
                DescriptorInfo<T2> tmp;
                for (auto item : descriptor)
                    tmp.push_back((T2)item);

                dst.push_back(tmp);
            }
        }
    }
}