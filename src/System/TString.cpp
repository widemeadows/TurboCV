#include "Core.h"

namespace TurboCV
{
namespace System
{
    // Copies the null-terminated character sequence (C-string) pointed by str.
    TString::TString(const char* str)
    {
        if (!str)
            throw ArgumentNullException();

        _length = std::strlen(str);

        _chars = new char[_length + 1];
        std::strcpy(_chars, str);
    }

    // Copies the std::string referenced by str.
    TString::TString(const std::string& str)
    {
        _length = str.length();

        _chars = new char[_length + 1];
        std::strcpy(_chars, str.c_str());
    }

    // Constructs a copy of str.
    TString::TString(const TString& str)
    {
        _length = str._length;

        _chars = new char[_length + 1];
        std::strcpy(_chars, str._chars);
    }

    // Assigns the null-terminated character sequence (C-string) to this instance, 
    // replacing its current contents.
    TString& TString::operator=(const char* str)
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
    TString& TString::operator=(const std::string& str)
    {
        _length = str.length();

        char* tmp = new char[_length + 1];
        std::strcpy(tmp, str.c_str());
        delete[] _chars;
        _chars = tmp;

        return *this;
    }

    // Assigns another TString object to this instance, replacing its current contents.
    TString& TString::operator=(const TString& str)
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
    TString TString::operator+(const char* append) const
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
    TString TString::operator+(const std::string& append) const
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
    TString TString::operator+(const TString& append) const
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
    TString operator+(const char* lhs, const TString& rhs)
    {
        return TString(lhs) + rhs;
    }

    // Returns a newly constructed TString object with its value being the concatenation of 
    // the characters in lhs followed by those of rhs.
    TString operator+(const std::string& lhs, const TString& rhs)
    {
        return TString(lhs) + rhs;
    }

    // Indicates whether this instance is less than str.
    bool TString::operator<(const TString& str) const
    {
        return std::strcmp(_chars, str._chars) < 0;
    }

    // Indicates whether this instance is equal to str.
    bool TString::operator==(const TString& str) const
    {
        return std::strcmp(_chars, str._chars) == 0;
    }

    // Inserts the sequence of characters that conforms value of str into out.
    std::ostream& operator<<(std::ostream& out, const TString& str)
    {
        out << str._chars;
        return out;
    }

    // Extracts a TString object from the input stream in, storing the sequence in str, 
    // which is overwritten (the previous value of str is deleted).
    std::istream& operator>>(std::istream& in, TString& str)
    {
        std::string tmp;
        in >> tmp;
        str = tmp;

        return in;
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and ends at the end of this instance.
    TString TString::Substring(size_t offset) const
    {
        return Substring(offset, _length - offset);
    }

    // Returns a newly constructed string object with its value initialized to a copy of 
    // a substring of this instance. The substring is the portion of this instance that starts 
    // at character position offset and spans length characters.
    TString TString::Substring(size_t offset, size_t length) const
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
    size_t TString::IndexOf(char value) const
    {
        char* ptr = std::strchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the first occurrence of substr.
    // If found, the index indicated the first occurrence of substr is returned. Otherwise, -1 is returned.
    size_t TString::IndexOf(const TString& substr) const
    {
        char* ptr = std::strstr(_chars, substr._chars);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last character that matches the characters specified in its arguments.
    // If found, the index of the specific character is returned. Otherwise, -1 is returned.
    size_t TString::LastIndexOf(char value) const
    {
        char* ptr = std::strrchr(_chars, value);

        if (!ptr)
            return -1;
        else
            return ptr - _chars;
    }

    // Searches this instance for the last occurrence of substr.
    // If found, the index indicated the last occurrence of substr is returned. Otherwise, -1 is returned.
    size_t TString::LastIndexOf(const TString& substr) const
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
    ArrayList<TString> TString::Split(char separateCharacter) const
    {
        char tmp[] = { separateCharacter };
        return Split(tmp);
    }

    // Returns a TString array that contains the substrings in this instance that are delimited 
    // by elements of a character array.
    ArrayList<TString> TString::Split(const char* separateCharacters) const
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
}
}