#pragma once

#include <exception>
#include <string>
#include "String.h"
using namespace std;

namespace System
{
	class SystemException : public exception
	{
	public:
		SystemException() : exception() {};
		SystemException(String message) : exception(message) {};
	};

	class ArgumentException : public SystemException
	{
	public:
		ArgumentException() : SystemException("Argument Error") {};
		ArgumentException(String message) : SystemException(message) {};
	};

	class ArgumentNullException : public ArgumentException
	{
	public:
		ArgumentNullException() : ArgumentException("Argument is Null") {};
		ArgumentNullException(String message) : ArgumentException(message) {};
	};

	class ArgumentOutOfRangeException : public ArgumentException
	{
	public:
		ArgumentOutOfRangeException() : ArgumentException("Argument is Out of Range") {};
		ArgumentOutOfRangeException(String message) : ArgumentException(message) {};
	};
}