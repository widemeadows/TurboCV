#pragma once
#pragma warning(disable:4996) // disable _CRT_SECURE_NO_WARNINGS

#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "windows.h"

namespace TurboCV
{
namespace System
{
    //////////////////////////////////////////////////////////////////////////
    // APIs for Group
    //////////////////////////////////////////////////////////////////////////

    class NullType {};

    template<typename T1 = NullType, typename T2 = NullType, 
        typename T3 = NullType, typename T4 = NullType>
    class Group
    {

    };

    template<typename T1, typename T2>
    class Group<T1, T2, NullType, NullType>
    {
    public:
        Group() {}
        Group(const T1& i1, const T2& i2) : item1(i1), item2(i2) {}

        T1& Item1() { return item1; }
        T2& Item2() { return item2; }
        const T1& Item1() const { return item1; }
        const T2& Item2() const { return item2; }

        bool operator<(const Group<T1, T2>& v) const
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
    Group<T1, T2> CreateGroup(const T1& i1, const T2& i2) 
    { 
        return Group<T1, T2>(i1, i2); 
    }

    template<typename T1, typename T2, typename T3>
    class Group<T1, T2, T3, NullType>
    {
    public:
        Group() {}
        Group(const T1& i1, const T2& i2, const T3& i3) : item1(i1), item2(i2), item3(i3) {}

        T1& Item1() { return item1; }
        T2& Item2() { return item2; }
        T3& Item3() { return item3; }
        const T1& Item1() const { return item1; }
        const T2& Item2() const { return item2; }
        const T3& Item3() const { return item3; }

        bool operator<(const Group<T1, T2, T3>& v) const
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
    Group<T1, T2, T3> CreateGroup(const T1& i1, const T2& i2, const T3& i3) 
    { 
        return Group<T1, T2, T3>(i1, i2, i3); 
    }


    //////////////////////////////////////////////////////////////////////////
    // APIs for Smart Pointer
    //////////////////////////////////////////////////////////////////////////

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


    //////////////////////////////////////////////////////////////////////////
    // APIs for ArrayList
    //////////////////////////////////////////////////////////////////////////

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


    //////////////////////////////////////////////////////////////////////////
    // APIs for TString
    //////////////////////////////////////////////////////////////////////////

    class TString
    {
    public:
        // Constructs an empty TString object, with a length of zero characters.
        TString();

        // Copies the null-terminated character sequence (C-string) pointed by str.
        TString(const char* str);

        // Copies the std::string referenced by str.
        TString(const std::string& str);

        // Constructs a copy of str.
        TString(const TString& str);

        // Destroys this instance.
        // This method deallocates all the storage capacity allocated by this instance using its allocator.
        ~TString()
        {
            delete[] _chars;
        }

        // Assigns the null-terminated character sequence (C-string) to this instance, 
        // replacing its current contents.
        TString& operator=(const char* str);

        // Assigns the std::string to this instance, replacing its current contents.
        TString& operator=(const std::string& str);

        // Assigns another TString object to this instance, replacing its current contents.
        TString& operator=(const TString& str);

        // Returns a newly constructed TString object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        TString operator+(const char* append) const;

        // Returns a newly constructed TString object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        TString operator+(const std::string& append) const;

        // Returns a newly constructed TString object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        TString operator+(const TString& append) const;

        // Returns a newly constructed TString object with its value being the concatenation of 
        // the characters in lhs followed by those of rhs.
        friend TString operator+(const char* lhs, const TString& rhs);

        // Returns a newly constructed TString object with its value being the concatenation of 
        // the characters in lhs followed by those of rhs.
        friend TString operator+(const std::string& lhs, const TString& rhs);

        // Indicates whether this instance is less than str.
        bool operator<(const TString& str) const;

        // Indicates whether this instance is equal to str.
        bool operator==(const TString& str) const;

        // Converts this instance to a null-terminated character sequence (C-string).
        // A pointer to an array that contains a null-terminated sequence of characters is returned.
        // However, different from Chars() method, the pointer just points to an internal structure
        // of this instance (shadow copy). Therefore, the contents in the array should not be modified.
        operator const char*() const;

        // Converts this instance to an std::string.
        operator std::string() const;

        // Inserts the sequence of characters that conforms value of str into out.
        friend std::ostream& operator<<(std::ostream& out, const TString& str);

        // Extracts a TString object from the input stream in, storing the sequence in str, 
        // which is overwritten (the previous value of str is deleted).
        friend std::istream& operator>>(std::istream& in, TString& str);

        // Returns a newly constructed string object with its value initialized to a copy of 
        // a substring of this instance. The substring is the portion of this instance that starts 
        // at character position offset and ends at the end of this instance.
        TString Substring(size_t offset) const;

        // Returns a newly constructed string object with its value initialized to a copy of 
        // a substring of this instance. The substring is the portion of this instance that starts 
        // at character position offset and spans length characters.
        TString Substring(size_t offset, size_t length) const;

        // Searches this instance for the first character that matches the characters specified in its arguments.
        // If found, the index of the specific character is returned. Otherwise, -1 is returned.
        size_t IndexOf(char value) const;

        // Searches this instance for the first occurrence of substr.
        // If found, the index indicated the first occurrence of substr is returned. Otherwise, -1 is returned.
        size_t IndexOf(const TString& substr) const;

        // Searches this instance for the last character that matches the characters specified in its arguments.
        // If found, the index of the specific character is returned. Otherwise, -1 is returned.
        size_t LastIndexOf(char value) const;

        // Searches this instance for the last occurrence of substr.
        // If found, the index indicated the last occurrence of substr is returned. Otherwise, -1 is returned.
        size_t LastIndexOf(const TString& substr) const;

        // Returns a TString array that contains the substrings in this instance that are delimited 
        // by the specific character.
        ArrayList<TString> Split(char separateCharacter) const;

        // Returns a TString array that contains the substrings in this instance that are delimited 
        // by elements of a character array.
        ArrayList<TString> Split(const char* separateCharacters) const;

        // Returns a pointer to an array that contains a null-terminated sequence of characters 
        // (i.e., a C-string) representing the current value of this instance.
        // The array mentioned above is newly allocated (i.e., deep copy), so after using this C-TString,
        // delete[] operation on the returned pointer is required.
        char* Chars() const;

        // Returns the length of this instance, in terms of number of characters.
        size_t Length() const;

    private:
        char* _chars;
        size_t _length;
    };

    // Constructs an empty TString object, with a length of zero characters.
    inline TString::TString()
    {
        _length = 0;

        _chars = new char[1];
        _chars[0] = '\0';
    }

    // Converts this instance to a null-terminated character sequence (C-string).
    // A pointer to an array that contains a null-terminated sequence of characters is returned.
    // However, different from Chars() method, the pointer just points to an internal structure
    // of this instance (shadow copy). Therefore, the contents in the array should not be modified.
    inline TString::operator const char*() const
    {
        return _chars;
    }

    // Converts this instance to an std::string.
    inline TString::operator std::string() const
    {
        return std::string(_chars); 
    }

    // Returns a pointer to an array that contains a null-terminated sequence of characters 
    // (i.e., a C-string) representing the current value of this instance.
    // The array mentioned above is newly allocated (i.e., deep copy), so after using this C-TString,
    // delete[] operation on the returned pointer is required.
    inline char* TString::Chars() const
    {
        char* result = new char[_length + 1];
        std::strcpy(result, _chars);

        return result;
    }

    // Returns the length of this instance, in terms of number of characters.
    inline size_t TString::Length() const
    {
        return _length;
    }

    
    //////////////////////////////////////////////////////////////////////////
    // APIs for Basic Type
    //////////////////////////////////////////////////////////////////////////

    class Int
    {
    public:
        static int Parse(const TString& integer)
        {
            std::stringstream ss;
            int result;

            ss << integer;
            ss >> result;

            return result;
        }

        static TString ToString(int integer)
        {
            std::stringstream ss;
            TString result;

            ss << integer;
            ss >> result;

            return result;
        }
    };

    class Float
    {
    public:
        static float Parse(const TString& floating)
        {
            std::stringstream ss;
            float result;

            ss << std::setprecision(7) << floating;
            ss >> result;

            return result;
        }

        static TString ToString(float floating)
        {
            std::stringstream ss;
            TString result;

            ss << std::setprecision(7) << floating;
            ss >> result;

            return result;
        }
    };

    class Double
    {
    public:
        static double Parse(const TString& floating)
        {
            std::stringstream ss;
            double result;

            ss << std::setprecision(16) << floating;
            ss >> result;

            return result;
        }

        static TString ToString(double floating)
        {
            std::stringstream ss;
            TString result;

            ss << std::setprecision(16) << floating;
            ss >> result;

            return result;
        }
    };


    //////////////////////////////////////////////////////////////////////////
    // APIs for Exception
    //////////////////////////////////////////////////////////////////////////

    class SystemException : public std::exception
    {
    public:
        SystemException() {};
        SystemException(const std::string& message) : _message(message) {};

        virtual const char* what() const { return _message.c_str(); };

    private:
        std::string _message;
    };

    class ArgumentException : public SystemException
    {
    public:
        ArgumentException() : SystemException("Argument Error") {};
        ArgumentException(const std::string& message) : SystemException(message) {};
    };

    class ArgumentNullException : public ArgumentException
    {
    public:
        ArgumentNullException() : ArgumentException("Argument is Null") {};
        ArgumentNullException(const std::string& message) : ArgumentException(message) {};
    };

    class ArgumentOutOfRangeException : public ArgumentException
    {
    public:
        ArgumentOutOfRangeException() : ArgumentException("Argument is Out of Range") {};
        ArgumentOutOfRangeException(const std::string& message) : ArgumentException(message) {};
    };


    //////////////////////////////////////////////////////////////////////////
    // APIs for File System
    //////////////////////////////////////////////////////////////////////////

    // Provides the base class for both FileInfo and DirectoryInfo objects.
    class FileSystemInfo
    {
    public:
        // Initializes a new instance of the FileSystemInfo class.
        FileSystemInfo(const TString& path);

        // Gets the string representing the extension part of the file.
        TString Extension() const;

        // For files, gets the name of the file. For directories, gets the name of 
        // the last directory in the hierarchy if a hierarchy exists. Otherwise, 
        // the Name method gets the name of the directory.
        TString Name() const;

        // Gets the full path of the directory or file.
        TString FullName() const;

        // Gets a value indicating whether the file or directory exists.
        virtual bool Exists() const = 0;

        // Creates a file or directory.
        virtual bool Create() const = 0;

        // Deletes a file or directory.
        virtual bool Delete() const = 0;

        // Moves a specified file / direcotry to a new location, 
        // providing the option to specify a new file / directory name.
        bool MoveTo(const TString& newPath);

    protected:
        TString _path;
    };

    // Provides instance methods for the creation, deletion and moving of files.
    class FileInfo : public FileSystemInfo
    {
    public:
        // Initializes a new instance of the FileInfo class, 
        // which acts as a wrapper for a file path.
        FileInfo(const TString& path) : FileSystemInfo(path) {};

        // Overridden. Gets a value indicating whether a file exists.
        virtual bool Exists() const;

        // Overrideen. Creates a file.
        virtual bool Create() const;

        // Overridden. Permanently deletes a file.
        virtual bool Delete() const;

        // Gets the full path of the parent directory.
        TString Directory() const;
    };

    // Provides instance methods for the creation, deletion and moving of directories. 
    // Also provides instance methods for generating file and subdirectory list.
    class DirectoryInfo : public FileSystemInfo
    {
    public:
        // Initializes a new instance of the DirectoryInfo class on the specified path.
        DirectoryInfo(const TString& path) : FileSystemInfo(path) {};

        // Overridden. Gets a value indicating whether the directory exists.
        virtual bool Exists() const;

        // Overridden. Creates a directory.
        virtual bool Create() const;

        // Overridden. Deletes a DirectoryInfo and its contents from a path.
        virtual bool Delete() const;

        // Gets the parent directory of a specified subdirectory.
        TString Parent() const;

        // Returns the full paths of subdirectories in the current directory.
        ArrayList<TString> GetDirectories() const;

        // Returns the full paths of files in the current directory.
        ArrayList<TString> GetFiles() const;
    };


    //////////////////////////////////////////////////////////////////////////
    // APIs for Random Selection
    //////////////////////////////////////////////////////////////////////////

    template<typename T>
    inline ArrayList<T> PickUp(const ArrayList<T>& vec, const ArrayList<size_t>& pickUpIndexes)
    {
        ArrayList<size_t> indexes = pickUpIndexes;
        std::sort(indexes.begin(), indexes.end());

        ArrayList<T> pickUps;
        size_t cardNum = vec.Count(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.Count() && indexes[counter] == i)
            {
                counter++;
                pickUps.Add(vec[i]);
            }
        }

        return pickUps;
    }

    template<typename T>
    inline Group<ArrayList<T>, ArrayList<T>> Divide(const ArrayList<T>& vec, const ArrayList<size_t>& pickUpIndexes)
    {
        ArrayList<size_t> indexes = pickUpIndexes;
        std::sort(indexes.begin(), indexes.end());

        ArrayList<T> pickUps, others;
        size_t cardNum = vec.Count(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.Count() && indexes[counter] == i)
            {
                counter++;
                pickUps.Add(vec[i]);
            }
            else
                others.Add(vec[i]);
        }

        return CreateGroup(pickUps, others);
    }

    inline ArrayList<size_t> RandomPermutate(size_t cardNum, size_t pickUpNum)
    {
        assert(cardNum >= pickUpNum);
        ArrayList<size_t> result;

        size_t* cards = new size_t[cardNum];
        for (size_t i = 0; i < cardNum; i++)
            cards[i] = i;

        for (size_t i = 0; i < pickUpNum; i++)
        {
            size_t index = (size_t)((double)rand() * (cardNum - i - 1) / RAND_MAX + i);
            assert(index < cardNum);
            std::swap(cards[i], cards[index]);
        }

        for (size_t i = 0; i < pickUpNum; i++)
            result.Add(cards[i]);

        delete[] cards;
        return result;
    }

    template<typename T>
    inline ArrayList<T> RandomPickUp(const ArrayList<T>& vec, size_t pickUpNum)
    {
        size_t cardNum = vec.Count();
        assert(cardNum >= pickUpNum);

        return PickUp(vec, RandomPermutate(cardNum, pickUpNum));
    }

    inline ArrayList<ArrayList<size_t>> RandomSplit(size_t cardNum, size_t foldNum)
    {
        assert(cardNum >= foldNum);

        ArrayList<size_t> permutation = RandomPermutate(cardNum, cardNum);

        ArrayList<ArrayList<size_t>> result;
        for (size_t i = 0; i < foldNum; i++)
        {
            ArrayList<size_t> subsetIndexes;
            size_t begin = cardNum / foldNum * i, 
                   end = (i != foldNum - 1) ? cardNum / foldNum * (i + 1) : cardNum;

            for (size_t j = begin; j < end; j++)
                subsetIndexes.Add(permutation[j]);

            std::sort(subsetIndexes.begin(), subsetIndexes.end());
            result.Add(subsetIndexes);
        }

        return result;
    }
}
}