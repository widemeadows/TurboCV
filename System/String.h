#pragma once

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
		String();
		String(const char* str);
		String(const string& str);
		String(const String& str);

		~String()
		{
			delete[] _chars;
		}

		String& operator=(const char* str);
		String& operator=(const string& str);
		String& operator=(const String& str);

		String operator+(const char* append) const;
		String operator+(const string& append) const;
		String operator+(const String& append) const;
		friend String operator+(const char* lhs, const String& rhs);
		friend String operator+(const string& lhs, const String& rhs);

		bool operator<(const String& str) const;
		bool operator==(const String& str) const;

		operator const char*() const;
		operator string() const;

		friend ostream& operator<<(ostream& out, const String& str);
		friend istream& operator>>(istream& in, String& str);

		String Substring(int offset) const;
		String Substring(int offset, int length) const;

		int IndexOf(char value) const;
		int IndexOf(const String& substr) const;

		vector<String> Split(char separateCharacter) const;
		vector<String> Split(const char* separateCharacters) const;

		char* Chars() const;
		int Length() const;
		
	private:
		char* _chars;
		int _length;
	};

	inline String::String()
	{
		_length = 0;

		_chars = new char[1];
		_chars[0] = '\0';
	}

	inline String::String(const char* str)
	{
		if (!str)
			throw ArgumentNullException();

		_length = strlen(str);

		_chars = new char[_length + 1];
		strcpy(_chars, str);
	}

	inline String::String(const string& str)
	{
		_length = str.length();

		_chars = new char[_length + 1];
		strcpy(_chars, str.c_str());
	}

	inline String::String(const String& str)
	{
		_length = str._length;

		_chars = new char[_length + 1];
		strcpy(_chars, str._chars);
	}

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

	inline String& String::operator=(const string& str)
	{
		_length = str.length();

		char* tmp = new char[_length + 1];
		strcpy(tmp, str.c_str());
		delete[] _chars;
		_chars = tmp;

		return *this;
	}

	inline String& String::operator=(const String& str)
	{
		_length = str._length;

		char* tmp = new char[_length + 1];
		strcpy(tmp, str._chars);
		delete[] _chars;
		_chars = tmp;

		return *this;
	}

	inline String String::operator+(const char* append) const
	{
		if (!append)
			throw ArgumentNullException();

		int appendLength = strlen(append);
		String result;

		result._length = _length + appendLength;
		result._chars = new char[result._length + 1];
		strcpy(result._chars, _chars);
		strcat(result._chars, append);

		return result;
	}

	inline String String::operator+(const string& append) const
	{
		int appendLength = append.length();
		String result;

		result._length = _length + appendLength;
		result._chars = new char[result._length + 1];
		strcpy(result._chars, _chars);
		strcat(result._chars, append.c_str());

		return result;
	}

	inline String String::operator+(const String& append) const
	{
		int appendLength = append._length;
		String result;

		result._length = _length + appendLength;
		result._chars = new char[result._length + 1];
		strcpy(result._chars, _chars);
		strcat(result._chars, append._chars);

		return result;
	}

	inline String operator+(const char* lhs, const String& rhs)
	{
		return String(lhs) + rhs;
	}

	inline String operator+(const string& lhs, const String& rhs)
	{
		return String(lhs) + rhs;
	}

	inline bool String::operator<(const String& str) const
	{
		return strcmp(_chars, str._chars) < 0;
	}

	inline bool String::operator==(const String& str) const
	{
		return strcmp(_chars, str._chars) == 0;
	}

	inline String::operator const char*() const
	{
		return _chars;
	}

	inline String::operator string() const
	{
		return string(_chars); 
	}

	inline ostream& operator<<(ostream& out, const String& str)
	{
		out << str._chars;
		return out;
	}

	inline istream& operator>>(istream& in, String& str)
	{
		string tmp;
		in >> tmp;
		str = tmp;

		return in;
	}

	inline String String::Substring(int offset) const
	{
		return Substring(offset, _length - offset);
	}

	inline String String::Substring(int offset, int length) const
	{
		if (offset < 0 || length < 0 || offset + length > _length)
			throw ArgumentOutOfRangeException();

		String result;
		result._length = length;
		result._chars = new char[length + 1];
		strncpy(result._chars, _chars + offset, length);
		result._chars[length] = '\0';

		return result;
	}

	inline int String::IndexOf(char value) const
	{
		char* ptr = strchr(_chars, value);

		if (!ptr)
			return -1;
		else
			return ptr - _chars;
	}

	inline int String::IndexOf(const String& substr) const
	{
		char* ptr = strstr(_chars, substr._chars);

		if (!ptr)
			return -1;
		else
			return ptr - _chars;
	}

	inline vector<String> String::Split(char separateCharacter) const
	{
		char tmp[] = { separateCharacter };
		return Split(tmp);
	}

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

	inline char* String::Chars() const
	{
		char* result = new char[_length + 1];
		strcpy(result, _chars);

		return result;
	}

	inline int String::Length() const
	{
		return _length;
	}
}