#ifndef TEXCEPTION_H
#define TEXCEPTION_H

#include <cstdint>

#include <stdexcept>
using std::exception;

// Contains filesystem::path, string
#include "Utility.h"

//refactor please

#define Throw(msg) throw(tException(msg, Canonical("/proc/self/exe").filename(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

struct tException: public exception
{
    inline tException(void) = delete;

    inline tException(const tException &message)
    {
        description = message.description;
        return;
    }

    inline tException(const string &message, const string &exe, const path &file, const uint32_t line, const string &function)
    {
        description = Format(R"("Program: "%s", File: "%s", Function: "%s", Line: %u, Exception: "%s")"s, exe.c_str(), file.c_str(), function.c_str(), line, message.c_str());
    }

    inline virtual ~tException(void) throw ()
    {
        return;
    }

    inline tException& operator=(const tException &execption) throw ()
    {
        description = execption.description;
        return *this;
    }

    inline virtual const char* what(void) const throw ()
    {
        return description.c_str();
    }

private:
    string description;
};

inline void throwException(const tException &exception)
{
    throw exception;
}

#endif
