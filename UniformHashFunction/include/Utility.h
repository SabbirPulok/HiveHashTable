#ifndef Utility_h
#define Utility_h

#include <cxxabi.h>

#include <pthread.h>
#include <sched.h>

#include <cstdarg>

#include <string>
  using std::string;
  using namespace std::literals;

#include <sstream>
using std::ostringstream;
using std::istringstream;

#include <vector>
  using std::vector;

#include <chrono>
  using HiResClock = std::chrono::high_resolution_clock;

#include <ctime>
  using std::gmtime;
  using std::localtime;
  using std::tm;

#include <iomanip>
  using std::put_time;
  using std::setw;
  using std::setfill;

#include "ForEach.h"
#include "pthreadErrors.h"

// ******************************************************************************************************
// This function implements a string printf function.  It's name and functionality matches the
// function implemented in the C++20 include <format>.
// ******************************************************************************************************
inline string
Format(const string &format, ...)
{
    va_list ap;

    // First get number of characters needed (less terminating null).
    va_start(ap, format);
    auto nChars { vsnprintf(nullptr, 0u, format.c_str(), ap) };
    va_end(ap);

    // Allocate buffer, including room for null.
    vector<char> Buffer(nChars + 1);

    // Print for real this time.
    va_start(ap, format);
    nChars = vsnprintf(Buffer.data(), Buffer.size(), format.c_str(), ap);
    va_end(ap);

    // Return resulting string.
    return string(Buffer.data());
}

//Convert a string to any numeric type with >> defined.
template<typename T>
inline T
ToNumber(const string &value)
{
    T temp;
    istringstream iss(value);
    iss >> temp;

    return temp;
}

// Convent any type with << defined to a string.
template<typename T>
inline string
ToString(const T &value)
{
    ostringstream oss;
    oss << value;

    return oss.str();
}

#define Break(cond) if (cond) {__asm__("int $3"); }
#define Assert(cond) if (!(cond)) {__asm__("int $3"); }

#define DumpVar(var) #var ":" << var
#define SepDumpVar(var) "," #var ": " << var

// Exposes the member variables of structs (under the assumption that a memberlist macro exists for said struct)
#define ExposeStruct(varType, var) auto &[varType##MemberList] { var }
#define ExposeStructConst(varType, var) const auto &[varType##MemberList] { var }

#define SepDumpListVar(var) << "," #var ":" << var
#define DumpListHelper(first, ...) DumpVar(first) _FOR_EACH(SepDumpListVar, __VA_ARGS__)
#define DumpList(x) DumpListHelper(x)

#define GenDumpStruct(Struct)                                      \
inline ostream &                                                   \
operator<<(ostream &os, const Struct &Var)                         \
{                                                                  \
  ExposeStructConst(Struct, Var);                                  \
  return os << "{" << Struct##DumpList << "}";                     \
}

#define GenDumpTemplateStruct(Struct)                              \
template<typename T>                                               \
inline ostream &                                                   \
operator<<(ostream &os, const Struct<T> &Var)                      \
{                                                                  \
  ExposeStructConst(Struct, Var);                                  \
  return os << "{" << Struct##DumpList << "}";                     \
}

inline path
Canonical(const path &filePath)
{
    if (exists(filePath))
    {
        // File exists.  We can return canonical form.
        return canonical(filePath);
    }
    else
    {
        // File does not exist.  the best we can do is absolute form.
        return absolute(filePath);
    }
}

// Get a copy of lower case string.
inline string
ToLower(const string &source)
{
    string temp(source.size(), ' ');
    transform(source.begin(), source.end(), temp.begin(), ::tolower);

    return temp;
}

// Lower case string in place.
inline string &
ToLowerThis(string &source)
{
    transform(source.begin(), source.end(), source.begin(), ::tolower);

    return source;
}

// Get a copy of Upper case string.
inline string
ToUpper(const string &source)
{
    string temp(source.size(), ' ');
    transform(source.begin(), source.end(), temp.begin(), ::toupper);

    return temp;
}

// Upper case string in place.
inline string &
ToUpperThis(string &source)
{
    transform(source.begin(), source.end(), source.begin(), ::toupper);

    return source;
}

inline string &
LeftTrimThis(string &SourceString, const string &MatchChars = " \t")
{
    SourceString.erase(0, SourceString.find_first_not_of(MatchChars));

    return SourceString;
}

inline string
LeftTrim(const string &SourceString, const string &MatchChars = " \t")
{
  string TempString { SourceString };

  return LeftTrimThis(TempString, MatchChars);
}

inline string &
RightTrimThis(string &SourceString, const string &MatchChars = " \t")
{
  SourceString.erase(SourceString.find_last_not_of(MatchChars) + 1u);

  return SourceString;
}

inline string
RightTrim(const string &SourceString, const string &MatchChars = " \t")
{
  string TempString { SourceString };

  return RightTrimThis(TempString, MatchChars);
}

inline string &
TrimThis(string &SourceString, const string &MatchChars = " \t")
{
  return LeftTrimThis(RightTrimThis(SourceString, MatchChars), MatchChars);
}

inline string
Trim(const string &SourceString, const string &MatchChars = " \t")
{
  string TempString { SourceString };

  return TrimThis(TempString, MatchChars);
}

inline string
TimestampLocal(void)
{
//  // Get now in high resolution.
  auto HiResNow = HiResClock::now();

  // Convert to time_t for easier conversion.
  auto NowTimeT = HiResClock::to_time_t(HiResNow);

  // Get gmtime and convert to a struct tm for formatting.
  tm *tm_now = localtime(&NowTimeT);

  // Extract the nanoseconds.
  auto Nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(HiResNow.time_since_epoch()).count() % 1'000'000'000;

  // Construct timestamp.
  ostringstream oss;
  oss << put_time(tm_now, "%Y-%m-%d %H:%M:%S")
            << "."s << setw(9) << setfill('0') << Nanoseconds // Print fractional seconds
            << put_time(tm_now, " (%Z)");

  // Return timestamp to caller.
  return oss.str();
}

inline string
TimestampUTC(void)
{
  // Get now in high resolution.
  auto HiResNow = HiResClock::now();

  // Convert to time_t for easier conversion.
  auto NowTimeT = HiResClock::to_time_t(HiResNow);

  // Get gmtime and convert to a struct tm for formatting.
  tm *tm_now = gmtime(&NowTimeT);

  // Extract the nanoseconds.
  auto Nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(HiResNow.time_since_epoch()).count() % 1'000'000'000;

  // Construct timestamp.
  ostringstream oss;
  oss << put_time(tm_now, "%Y-%m-%d %H:%M:%S")
            << "."s << setw(9) << setfill('0') << Nanoseconds // Print fractional seconds
            << " (UTC)"s;

  // Return timestamp to caller.
  return oss.str();
}

inline void
Timestamp(string &LocalTimestamp, string &UTCTimestamp)
{
  // Get now in high resolution.
  auto HiResNow = HiResClock::now();

  // Convert to time_t for easier conversion.
  auto NowTimeT = HiResClock::to_time_t(HiResNow);

  {
    // Get gmtime and convert to a struct tm for formatting.
    tm *tm_now = gmtime(&NowTimeT);

    // Extract the nanoseconds.
    auto Nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(HiResNow.time_since_epoch()).count() % 1'000'000'000;

    // Construct timestamp.
    ostringstream oss;
    oss << put_time(tm_now, "%Y-%m-%d %H:%M:%S")
              << "."s << setw(9) << setfill('0') << Nanoseconds // Print fractional seconds
              << " (UTC)"s;

    // Return timestamp to caller.
    UTCTimestamp = oss.str();
  }

  {
    // Get gmtime and convert to a struct tm for formatting.
    tm *tm_now = localtime(&NowTimeT);

    // Extract the nanoseconds.
    auto Nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(HiResNow.time_since_epoch()).count() % 1'000'000'000;

    // Construct timestamp.
    ostringstream oss;
    oss << put_time(tm_now, "%Y-%m-%d %H:%M:%S")
              << "."s << setw(9) << setfill('0') << Nanoseconds // Print fractional seconds
              << put_time(tm_now, " (%Z)");

    // Return timestamp to caller.
    LocalTimestamp = oss.str();
  }

  return;
}

inline void
SetThreadCPUAffinity(const vector<unsigned int> CPUs, pthread_t thread = pthread_self())
{
  cpu_set_t cpuset;
  // Initialize so that no CPUs are selected in the affinity mask..
  CPU_ZERO(&cpuset);

  for (const auto &CPU : CPUs)
  {
    CPU_SET(CPU, &cpuset);
  }

  // Set the new cpu affinity mask for this thread.
  DecodePThreadError(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset));

  return;
}

#define GetType(Var) Demangle(typeid(Var).name())
inline const string
Demangle(const string &Name) {

    int rc { -1 };

    char* Result = abi::__cxa_demangle(Name.c_str(), NULL, NULL, &rc);

    string DemangledName { (rc == 0) ? Result : Name };

    free(Result);

    return DemangledName;
}

#endif
