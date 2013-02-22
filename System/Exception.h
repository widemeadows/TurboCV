#pragma once

#include <exception>
using namespace std;

namespace System
{
	class String;

	class SystemException : public exception
	{
	public:
		SystemException() : exception() {};
		SystemException(const String& message) : exception(message) {};
	};

	class ArgumentException : public SystemException
	{
	public:
		ArgumentException() : SystemException("Argument Error") {};
		ArgumentException(const String& message) : SystemException(message) {};
	};

	class ArgumentNullException : public ArgumentException
	{
	public:
		ArgumentNullException() : ArgumentException("Argument is Null") {};
		ArgumentNullException(const String& message) : ArgumentException(message) {};
	};

	class ArgumentOutOfRangeException : public ArgumentException
	{
	public:
		ArgumentOutOfRangeException() : ArgumentException("Argument is Out of Range") {};
		ArgumentOutOfRangeException(const String& message) : ArgumentException(message) {};
	};
}