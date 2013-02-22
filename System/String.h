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
		inline String() : _chars(NULL), _length(0) {}
		explicit inline String(const char* str);
		explicit inline String(const string& str);
		inline String(const String& str);

		inline ~String()
		{
			delete[] _chars;
			_chars = NULL;
			_length = 0;
		}

		inline String& operator=(const char* str);
		inline String& operator=(const string& str);
		inline String& operator=(const String& str);

		inline String operator+(const char* append) const;
		inline String operator+(const string& append) const;
		inline String operator+(const String& append) const;
		inline friend String operator+(const char* lhs, const String& rhs);
		inline friend String operator+(const string& lhs, const String& rhs);

		inline bool operator<(const String& str) const;
		inline bool operator==(const String& str) const;

		inline operator char*() const;
		inline operator string() const;

		inline String Substring(int offset) const;
		inline String Substring(int offset, int length) const;

		inline int IndexOf(char value) const;
		inline int IndexOf(const String& substr) const;

		inline vector<String> Split(const char* separateCharacters) const;

		inline char* Chars() const;
		inline int Length() const;
		
		inline friend ostream& operator<<(ostream& out, const String& str);
		inline friend istream& operator>>(istream& in, String& str);

	private:
		char* _chars;
		int _length;
	};

	inline String::String(const char* str)
	{
		if (!str)
			_length = 0;
		else
			_length = strlen(str);

		if (!_length)
			_chars = NULL;
		else
		{
			_chars = new char[_length + 1];
			strcpy(_chars, str);
		}
	}

	inline String::String(const string& str)
	{
		_length = str.length();

		if (!_length)
			_chars = NULL;
		else
		{
			_chars = new char[_length + 1];
			strcpy(_chars, str.c_str());
		}
	}

	inline String::String(const String& str)
	{
		_length = str._length;

		if (!_length)
			_chars = NULL;
		else
		{
			_chars = new char[_length + 1];
			strcpy(_chars, str._chars);
		}
	}

	inline String& String::operator=(const char* str)
	{
		if (!str)
			_length = 0;
		else
			_length = strlen(str);

		if (!_length)
		{
			delete[] _chars;
			_chars = NULL;
		}
		else
		{
			char* tmp = new char[_length + 1];
			strcpy(tmp, str);
			delete[] _chars;
			_chars = tmp;
		}

		return *this;
	}

	inline String& String::operator=(const string& str)
	{
		_length = str.length();

		if (!_length)
		{
			delete[] _chars;
			_chars = NULL;
		}
		else
		{
			char* tmp = new char[_length + 1];
			strcpy(tmp, str.c_str());
			delete[] _chars;
			_chars = tmp;
		}

		return *this;
	}

	inline String& String::operator=(const String& str)
	{
		_length = str._length;

		if (!_length)
		{
			delete[] _chars;
			_chars = NULL;
		}
		else
		{
			char* tmp = new char[_length + 1];
			strcpy(tmp, str._chars);
			delete[] _chars;
			_chars = tmp;
		}

		return *this;
	}

	inline String String::operator+(const char* append) const
	{
		int appendLength = 0;

		if (!append)
			appendLength = 0;
		else
			appendLength = strlen(append);

		if (!appendLength)
			return *this;
		else
		{
			String result;

			result._length = _length + appendLength;
			result._chars = new char[result._length + 1];
			if (!_chars)
				strcpy(result._chars, append);
			else
			{
				strcpy(result._chars, _chars);
				strcat(result._chars, append);
			}

			return result;
		}
	}

	inline String String::operator+(const string& append) const
	{
		int appendLength = append.length();

		if (!appendLength)
			return *this;
		else
		{
			String result;

			result._length = _length + appendLength;
			result._chars = new char[result._length + 1];
			if (!_chars)
				strcpy(result._chars, append.c_str());
			else
			{
				strcpy(result._chars, _chars);
				strcat(result._chars, append.c_str());
			}

			return result;
		}
	}

	inline String String::operator+(const String& append) const
	{
		int appendLength = append._length;

		if (!appendLength)
			return *this;
		else
		{
			String result;

			result._length = _length + appendLength;
			result._chars = new char[result._length + 1];
			if (!_chars)
				strcpy(result._chars, append._chars);
			else
			{
				strcpy(result._chars, _chars);
				strcat(result._chars, append._chars);
			}

			return result;
		}
	}

	inline char* String::Chars() const
	{
		if (!_length)
			return NULL;

		char* result = new char[_length + 1];
		strcpy(result, _chars);

		return result;
	}

	inline int String::Length() const
	{
		return _length;
	}

	inline String operator+(const char* lhs, const String& rhs)
	{
		return rhs + lhs;
	}

	inline String operator+(const string& lhs, const String& rhs)
	{
		return rhs + lhs;
	}

	inline bool String::operator<(const String& str) const
	{
		if (!_chars)
		{
			if (!str._chars)
				return false;
			else
				return true;
		}
		else if (!str._chars)
			return false;
		else
			return strcmp(_chars, str._chars) < 0;
	}

	inline bool String::operator==(const String& str) const
	{
		if (!_chars || !str._chars)
		{
			if (!_chars && !str._chars)
				return true;
			else
				return false;
		}

		return strcmp(_chars, str._chars) == 0;
	}

	inline String::operator char*() const
	{
		if (!_chars)
			return NULL;
		else
		{
			char* result = new char[_length];
			strcpy(result, _chars);
			return result;
		}
	}

	inline String::operator string() const
	{
		if (!_chars)
			return string();
		else
			return string(_chars); 
	}

	inline String String::Substring(int offset) const
	{
		return Substring(offset, _length - offset);
	}

	inline String String::Substring(int offset, int length) const
	{
		if (offset < 0 || length == 0 || offset + length > _length)
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
		if (substr._chars == NULL)
			return 0;

		char* ptr = strstr(_chars, substr._chars);

		if (!ptr)
			return -1;
		else
			return ptr - _chars;
	}

	inline vector<String> String::Split(const char* separateCharacters) const
	{
		if (separateCharacters == NULL)
			throw ArgumentNullException();

		vector<String> tokens;
		int lastPos = -1, separatorNum = strlen(separateCharacters);

		for (int i = 0; i < _length; i++)
		{
			bool needToSplit = false;

			for (int j = 0; j < separatorNum; j++)
			{
				if (separateCharacters[j] == _chars[i])
				{
					needToSplit = true;
					break;
				}
			}

			if (needToSplit || i == _length - 1)
			{
				tokens.push_back(Substring(lastPos + 1, i - lastPos));
				lastPos = i;
			}
		}

		return tokens;
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
}