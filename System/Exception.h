#pragma once

#include <exception>
#include <string>
using namespace std;

namespace TurboCV
{
namespace System
{
    class SystemException : public exception
    {
    public:
        SystemException() {};
        SystemException(const string& message) : _message(message) {};

        virtual const char* what() const { return _message.c_str(); };

    private:
        string _message;
    };

    class ArgumentException : public SystemException
    {
    public:
        ArgumentException() : SystemException("Argument Error") {};
        ArgumentException(const string& message) : SystemException(message) {};
    };

    class ArgumentNullException : public ArgumentException
    {
    public:
        ArgumentNullException() : ArgumentException("Argument is Null") {};
        ArgumentNullException(const string& message) : ArgumentException(message) {};
    };

    class ArgumentOutOfRangeException : public ArgumentException
    {
    public:
        ArgumentOutOfRangeException() : ArgumentException("Argument is Out of Range") {};
        ArgumentOutOfRangeException(const string& message) : ArgumentException(message) {};
    };
}
}