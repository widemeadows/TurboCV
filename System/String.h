#pragma once
#pragma warning (disable: 4996)

#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include "Exception.h"
using namespace std;

namespace System
{
    class String
    {
    public:
        // Constructs an empty String object, with a length of zero characters.
        String();

        // Copies the null-terminated character sequence (C-string) pointed by str.
        String(const char* str);

        // Copies the std::string referenced by str.
        String(const string& str);

        // Constructs a copy of str.
        String(const String& str);

        // Destroys this instance.
        // This method deallocates all the storage capacity allocated by this instance using its allocator.
        ~String()
        {
            delete[] _chars;
        }

        // Assigns the null-terminated character sequence (C-string) to this instance, 
        // replacing its current contents.
        String& operator=(const char* str);

        // Assigns the std::string to this instance, replacing its current contents.
        String& operator=(const string& str);

        // Assigns another String object to this instance, replacing its current contents.
        String& operator=(const String& str);

        // Returns a newly constructed String object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        String operator+(const char* append) const;

        // Returns a newly constructed String object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        String operator+(const string& append) const;

        // Returns a newly constructed String object with its value being the concatenation of 
        // the characters in this instance followed by those of append.
        String operator+(const String& append) const;

        // Returns a newly constructed String object with its value being the concatenation of 
        // the characters in lhs followed by those of rhs.
        friend String operator+(const char* lhs, const String& rhs);

        // Returns a newly constructed String object with its value being the concatenation of 
        // the characters in lhs followed by those of rhs.
        friend String operator+(const string& lhs, const String& rhs);

        // Indicates whether this instance is less than str.
        bool operator<(const String& str) const;

        // Indicates whether this instance is equal to str.
        bool operator==(const String& str) const;

        // Converts this instance to a null-terminated character sequence (C-string).
        // A pointer to an array that contains a null-terminated sequence of characters is returned.
        // However, different from Chars() method, the pointer just points to an internal structure
        // of this instance (shadow copy). Therefore, the contents in the array should not be modified.
        operator const char*() const;

        // Converts this instance to an std::string.
        operator string() const;

        // Inserts the sequence of characters that conforms value of str into out.
        friend ostream& operator<<(ostream& out, const String& str);

        // Extracts a String object from the input stream in, storing the sequence in str, 
        // which is overwritten (the previous value of str is deleted).
        friend istream& operator>>(istream& in, String& str);

        // Returns a newly constructed string object with its value initialized to a copy of 
        // a substring of this instance. The substring is the portion of this instance that starts 
        // at character position offset and ends at the end of this instance.
        String Substring(size_t offset) const;

        // Returns a newly constructed string object with its value initialized to a copy of 
        // a substring of this instance. The substring is the portion of this instance that starts 
        // at character position offset and spans length characters.
        String Substring(size_t offset, size_t length) const;

        // Searches this instance for the first character that matches the characters specified in its arguments.
        // If found, the index of the specific character is returned. Otherwise, -1 is returned.
        size_t IndexOf(char value) const;

        // Searches this instance for the first occurrence of substr.
        // If found, the index indicated the first occurrence of substr is returned. Otherwise, -1 is returned.
        size_t IndexOf(const String& substr) const;

        // Searches this instance for the last character that matches the characters specified in its arguments.
        // If found, the index of the specific character is returned. Otherwise, -1 is returned.
        size_t LastIndexOf(char value) const;

        // Searches this instance for the last occurrence of substr.
        // If found, the index indicated the last occurrence of substr is returned. Otherwise, -1 is returned.
        size_t LastIndexOf(const String& substr) const;

        // Returns a String array that contains the substrings in this instance that are delimited 
        // by the specific character.
        vector<String> Split(char separateCharacter) const;

        // Returns a String array that contains the substrings in this instance that are delimited 
        // by elements of a character array.
        vector<String> Split(const char* separateCharacters) const;

        // Returns a pointer to an array that contains a null-terminated sequence of characters 
        // (i.e., a C-string) representing the current value of this instance.
        // The array mentioned above is newly allocated (i.e., deep copy), so after using this C-String,
        // delete[] operation on the returned pointer is required.
        char* Chars() const;

        // Returns the length of this instance, in terms of number of characters.
        size_t Length() const;
        
    private:
        char* _chars;
        size_t _length;
    };

    // Constructs an empty String object, with a length of zero characters.
    inline String::String()
    {
        _length = 0;

        _chars = new char[1];
        _chars[0] = '\0';
    }

    // Copies the null-terminated character sequence (C-string) pointed by str.
    inline String::String(const char* str)
    {
        if (!str)
            throw ArgumentNullException();

        _length = strlen(str);

        _chars = new char[_length + 1];
        strcpy(_chars, str);
    }

    // Copies the std::string referenced by str.
    inline String::String(const string& str)
    {
        _length = str.length();

        _chars = new char[_length + 1];
        strcpy(_chars, str.c_str());
    }

    // Constructs a copy of str.
    inline String::String(const String& str)
    {
        _length = str._length;

        _chars = new char[_length + 1];
        strcpy(_chars, str._chars);
    }

    // Assigns the null-terminated character sequence (C-string) to this instance, 
    // replacing its current contents.
    inline String& String::operator=(const char* str)
    {
        if (!str)
            throw ArgumentNullException();

        _length = strlen(str);

        char* tmp = new char[_length + 1];
        strcpy(tmp, str);
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Assigns the std::string to this instance, replacing its current contents.
    inline String& String::operator=(const string& str)
    {
        _length = str.length();

        char* tmp = new char[_length + 1];
        strcpy(tmp, str.c_str());
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Assigns another String object to this instance, replacing its current contents.
    inline String& String::operator=(const String& str)
    {
        _length = str._length;

        char* tmp = new char[_length + 1];
        strcpy(tmp, str._chars);
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Returns a newly constructed String object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline String String::operator+(const char* append) const
    {
        if (!append)
            throw ArgumentNullException();

        size_t appendLength = strlen(append);
        String result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        strcpy(result._chars, _chars);
        strcat(result._chars, append);

        return result;
    }

    // Returns a newly constructed String object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline String String::operator+(const string& append) const
    {
        size_t appendLength = append.length();
        String result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        strcpy(result._chars, _chars);
        strcat(result._chars, append.c_str());

        return result;
    }

    // Returns a newly constructed String object with its value being the concatenation of 
    // the characters in this instance followed by those of append.
    inline String String::operator+(const String& append) const
    {
        size_t appendLength = append._length;
        String result;

        result._length = _length + appendLength;
        result._chars = new char[result._length + 1];
        strcpy(result._chars, _chars);
        strcat(result._chars, append._chars);

        return result;
    }

    // Returns a newly constructed String object with its value being the concatenation of 
    // the characters in lhs followed by those of rhs.
    inline String operator+(const char* lhs, const String& rhs)
    {
        return String(lhs) + rhs;
    }

    // Returns a newly constructed String object with its value being the concatenation of 
    // the characters in lhs followed by those of rhs.
    inline String operator+(const string& lhs, const String& rhs)
    {
        return String(lhs) + rhs;
    }

    // Indicates whether this instance is less than str.
    inline bool String::operator<(const String& str) const
    {
        return strcmp(_chars, str._chars) < 0;
    }

    // Indicates whether this instance is equal to str.
    inline bool String::operator==(const String& str) const
    {
        return strcmp(_chars, str._chars) == 0;
    }

    // Converts this instance to a null-terminated character sequence (C-string).
    // A pointer to an array that contains a null-terminated sequence of characters is returned.
    // However, different from Chars() method, the pointer just points to an internal structure
    // of this instance (shadow copy). Therefore, the contents in the array should not be modified.
    inline String::operator const char*() const
    {
        return _chars;
    }

    // Converts this instance to an std::string.
    inline String::operator string() const
    {
        return string(_chars); 
    }

    // Inserts the sequence of characters that conforms value of str into out.
    inline ostream& operator<<(ostream& out, const String& str)
    {
        out << str._chars;
        return out;
    }

    // Extracts a String object from the input stream in, storing the sequence in str, 
    // which is overwritten (the previous value of str is deleted).
    inline istream& operator>>(istream& in, String& str)
    {
        string tmp;
        in >> tmp;
        str = tmp;

        return in;
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and ends at the end of this instance.
    inline String String::Substring(size_t offset) const
    {
        return Substring(offset, _length - offset);
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and spans length characters.
    inline String String::Substring(size_t offset, size_t length) const
    {
        if (offset + length > _length)
            throw ArgumentOutOfRangeException();

        String result;
        result._length = length;
        result._chars = new char[length + 1];
        strncpy(result._chars, _chars + offset, length);
        result._chars[length] = '\0';

        return result;
    }

    // Searches this instance for the first character that matches the characters specified in its arguments.
    // If found, the index of the specific character is returned. Otherwise, -1 is returned.
    inline size_t String::IndexOf(char value) const
    {
        char* ptr = strchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the first occurrence of substr.
    // If found, the index indicated the first occurrence of substr is returned. Otherwise, -1 is returned.
    inline size_t String::IndexOf(const String& substr) const
    {
        char* ptr = strstr(_chars, substr._chars);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last character that matches the characters specified in its arguments.
    // If found, the index of the specific character is returned. Otherwise, -1 is returned.
    inline size_t String::LastIndexOf(char value) const
    {
        char* ptr = strrchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last occurrence of substr.
    // If found, the index indicated the last occurrence of substr is returned. Otherwise, -1 is returned.
    inline size_t String::LastIndexOf(const String& substr) const
    {
        char* prev = NULL;
        char* cur = _chars;
        
        while (cur = strstr(cur, substr._chars))
        {
            prev = cur;
            cur++;
        }

        if (!prev)
            return -1;
        else
            return prev - _chars;
    }

    // Returns a String array that contains the substrings in this instance that are delimited 
    // by the specific character.
    inline vector<String> String::Split(char separateCharacter) const
    {
        char tmp[] = { separateCharacter };
        return Split(tmp);
    }

    // Returns a String array that contains the substrings in this instance that are delimited 
    // by elements of a character array.
    inline vector<String> String::Split(const char* separateCharacters) const
    {
        if (!separateCharacters)
            throw ArgumentNullException();

        vector<String> tokens;
        int beginPos = -1, endPos = -1;

        while (++beginPos < _length)
        {            
            if (!strchr(separateCharacters, _chars[beginPos]))
            {
                endPos = beginPos;
                
                while (++endPos < _length)
                {
                    if (strchr(separateCharacters, _chars[endPos]))
                        break;
                }

                tokens.push_back(Substring(beginPos, endPos - beginPos));
                beginPos = endPos;
            }
        }

        return tokens;
    }

    // Returns a pointer to an array that contains a null-terminated sequence of characters 
    // (i.e., a C-string) representing the current value of this instance.
    // The array mentioned above is newly allocated (i.e., deep copy), so after using this C-String,
    // delete[] operation on the returned pointer is required.
    inline char* String::Chars() const
    {
        char* result = new char[_length + 1];
        strcpy(result, _chars);

        return result;
    }

    // Returns the length of this instance, in terms of number of characters.
    inline size_t String::Length() const
    {
        return _length;
    }
}