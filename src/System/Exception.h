#pragma once

#include <exception>
#include <string>

namespace TurboCV
{
namespace System
{
    class SystemException : public std::exception
    {
    public:
        SystemException() {};
        SystemException(const std::string& message) : _message(message) {};

        virtual const char* what() const { return _message.c_str(); };

    private:
        std::string _message;
    };

    class ArgumentException : public SystemException
    {
    public:
        ArgumentException() : SystemException("Argument Error") {};
        ArgumentException(const std::string& message) : SystemException(message) {};
    };

    class ArgumentNullException : public ArgumentException
    {
    public:
        ArgumentNullException() : ArgumentException("Argument is Null") {};
        ArgumentNullException(const std::string& message) : ArgumentException(message) {};
    };

    class ArgumentOutOfRangeException : public ArgumentException
    {
    public:
        ArgumentOutOfRangeException() : ArgumentException("Argument is Out of Range") {};
        ArgumentOutOfRangeException(const std::string& message) : ArgumentException(message) {};
    };
}
}