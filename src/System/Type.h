#pragma once
#pragma warning(disable:4996) // disable _CRT_SECURE_NO_WARNINGS

#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Collection.h"
#include "Exception.h"

namespace TurboCV
{
namespace System
{
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

    // Copies the null-terminated character sequence (C-string) pointed by str.
    inline TString::TString(const char* str)
    {
        if (!str)
            throw ArgumentNullException();

        _length = std::strlen(str);

        _chars = new char[_length + 1];
        std::strcpy(_chars, str);
    }

    // Copies the std::string referenced by str.
    inline TString::TString(const std::string& str)
    {
        _length = str.length();

        _chars = new char[_length + 1];
        std::strcpy(_chars, str.c_str());
    }

    // Constructs a copy of str.
    inline TString::TString(const TString& str)
    {
        _length = str._length;

        _chars = new char[_length + 1];
        std::strcpy(_chars, str._chars);
    }

    // Assigns the null-terminated character sequence (C-string) to this instance, 
    // replacing its current contents.
    inline TString& TString::operator=(const char* str)
    {
        if (!str)
            throw ArgumentNullException();

        _length = std::strlen(str);

        char* tmp = new char[_length + 1];
        std::strcpy(tmp, str);
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Assigns the std::string to this instance, replacing its current contents.
    inline TString& TString::operator=(const std::string& str)
    {
        _length = str.length();

        char* tmp = new char[_length + 1];
        std::strcpy(tmp, str.c_str());
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Assigns another TString object to this instance, replacing its current contents.
    inline TString& TString::operator=(const TString& str)
    {
        _length = str._length;

        char* tmp = new char[_length + 1];
        std::strcpy(tmp, str._chars);
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline TString TString::operator+(const char* append) const
    {
        if (!append)
            throw ArgumentNullException();

        size_t appendLength = std::strlen(append);
        TString result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        std::strcpy(result._chars, _chars);
        std::strcat(result._chars, append);

        return result;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline TString TString::operator+(const std::string& append) const
    {
        size_t appendLength = append.length();
        TString result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        std::strcpy(result._chars, _chars);
        std::strcat(result._chars, append.c_str());

        return result;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline TString TString::operator+(const TString& append) const
    {
        size_t appendLength = append._length;
        TString result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        std::strcpy(result._chars, _chars);
        std::strcat(result._chars, append._chars);

        return result;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in lhs followed by those of rhs.
    inline TString operator+(const char* lhs, const TString& rhs)
    {
        return TString(lhs) + rhs;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in lhs followed by those of rhs.
    inline TString operator+(const std::string& lhs, const TString& rhs)
    {
        return TString(lhs) + rhs;
    }

    // Indicates whether this instance is less than str.
    inline bool TString::operator<(const TString& str) const
    {
        return std::strcmp(_chars, str._chars) < 0;
    }

    // Indicates whether this instance is equal to str.
    inline bool TString::operator==(const TString& str) const
    {
        return std::strcmp(_chars, str._chars) == 0;
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

    // Inserts the sequence of characters that conforms value of str into out.
    inline std::ostream& operator<<(std::ostream& out, const TString& str)
    {
        out << str._chars;
        return out;
    }

    // Extracts a TString object from the input stream in, storing the sequence in str, 
    // which is overwritten (the previous value of str is deleted).
    inline std::istream& operator>>(std::istream& in, TString& str)
    {
        std::string tmp;
        in >> tmp;
        str = tmp;

        return in;
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and ends at the end of this instance.
    inline TString TString::Substring(size_t offset) const
    {
        return Substring(offset, _length - offset);
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and spans length characters.
    inline TString TString::Substring(size_t offset, size_t length) const
    {
        if (offset + length > _length)
            throw ArgumentOutOfRangeException();

        TString result;
        result._length = length;
        result._chars = new char[length + 1];
        std::strncpy(result._chars, _chars + offset, length);
        result._chars[length] = '\0';

        return result;
    }

    // Searches this instance for the first character that matches the characters specified in its arguments.
    // If found, the index of the specific character is returned. Otherwise, -1 is returned.
    inline size_t TString::IndexOf(char value) const
    {
        char* ptr = std::strchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the first occurrence of substr.
    // If found, the index indicated the first occurrence of substr is returned. Otherwise, -1 is returned.
    inline size_t TString::IndexOf(const TString& substr) const
    {
        char* ptr = std::strstr(_chars, substr._chars);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last character that matches the characters specified in its arguments.
    // If found, the index of the specific character is returned. Otherwise, -1 is returned.
    inline size_t TString::LastIndexOf(char value) const
    {
        char* ptr = std::strrchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last occurrence of substr.
    // If found, the index indicated the last occurrence of substr is returned. Otherwise, -1 is returned.
    inline size_t TString::LastIndexOf(const TString& substr) const
    {
        char* prev = NULL;
        char* cur = _chars;

        while (cur = std::strstr(cur, substr._chars))
        {
            prev = cur;
            cur++;
        }

        if (!prev)
            return -1;
        else
            return prev - _chars;
    }

    // Returns a TString array that contains the substrings in this instance that are delimited 
    // by the specific character.
    inline ArrayList<TString> TString::Split(char separateCharacter) const
    {
        char tmp[] = { separateCharacter };
        return Split(tmp);
    }

    // Returns a TString array that contains the substrings in this instance that are delimited 
    // by elements of a character array.
    inline ArrayList<TString> TString::Split(const char* separateCharacters) const
    {
        if (!separateCharacters)
            throw ArgumentNullException();

        ArrayList<TString> tokens;
        int beginPos = -1, endPos = -1;

        while (++beginPos < _length)
        {            
            if (!std::strchr(separateCharacters, _chars[beginPos]))
            {
                endPos = beginPos;

                while (++endPos < _length)
                {
                    if (std::strchr(separateCharacters, _chars[endPos]))
                        break;
                }

                tokens.Add(Substring(beginPos, endPos - beginPos));
                beginPos = endPos;
            }
        }

        return tokens;
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

    ///////////////////////////////////////////////////////////////////////

    class Int
    {
    public:
        static int Parse(const TString& integer);
        static TString ToString(int integer);
    };

    inline int Int::Parse(const TString& integer)
    {
        std::stringstream ss;
        int result;

        ss << integer;
        ss >> result;

        return result;
    }

    inline TString Int::ToString(int integer)
    {
        std::stringstream ss;
        TString result;

        ss << integer;
        ss >> result;

        return result;
    }

    ///////////////////////////////////////////////////////////////////////

    class Float
    {
    public:
        static float Parse(const TString& floating);
        static TString ToString(float floating);
    };

    inline float Float::Parse(const TString& floating)
    {
        std::stringstream ss;
        float result;

        ss << std::setprecision(7) << floating;
        ss >> result;

        return result;
    }

    inline TString Float::ToString(float floating)
    {
        std::stringstream ss;
        TString result;

        ss << std::setprecision(7) << floating;
        ss >> result;

        return result;
    }

    ///////////////////////////////////////////////////////////////////////

    class Double
    {
    public:
        static double Parse(const TString& floating);
        static TString ToString(double floating);
    };

    inline double Double::Parse(const TString& floating)
    {
        std::stringstream ss;
        double result;

        ss << std::setprecision(16) << floating;
        ss >> result;

        return result;
    }

    inline TString Double::ToString(double floating)
    {
        std::stringstream ss;
        TString result;

        ss << std::setprecision(16) << floating;
        ss >> result;

        return result;
    }

    ///////////////////////////////////////////////////////////////////////

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
}
}