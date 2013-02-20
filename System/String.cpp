#include "String.h"

namespace System
{
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

	inline char* String::Chars()
	{
		if (!_length)
			return NULL;

		char* result = new char[_length + 1];
		strcpy(result, _chars);

		return result;
	}

	inline int String::Length()
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
}
