#include <cstring>
#include <string>
#include <iostream>
using namespace std;

namespace System
{
	class String
	{
	public:
		inline String() : _chars(NULL), _length(0) {}
		inline String(const char* str);
		inline String(const string& str);
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

		inline char* Chars();
		inline int Length();

	private:
		char* _chars;
		int _length;
	};
}