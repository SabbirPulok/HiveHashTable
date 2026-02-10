#ifndef pthreadErrors_h
#define pthreadErrors_h

#include <iostream>
using std::cout;  using std::cerr;  using std::endl;

#include <string>
  using std::string;
  using namespace std::literals;

#include <sstream>
using std::ostringstream;
using std::istringstream;

inline void
DecodePThreadError(int err)
{
  string erc;

  if (err == 0)
  {
    return;
  }

  switch (err)
  {
    case EPERM:
    {
      erc = "EPERM - Operation not permitted"s;
      break;
    }

    case ENOENT:
    {
      erc = "ENOENT - No such file or directory"s;
      break;
    }

    case ESRCH:
    {
      erc = "ESRCH - No such process"s;
      break;
    }

    case EINTR:
    {
      erc = "EINTR - Interrupted system call"s;
      break;
    }

    case EIO:
    {
      erc = "EIO - I/O error"s;
      break;
    }

    case ENXIO:
    {
      erc = "ENXIO - No such device or address"s;
      break;
    }

    case E2BIG:
    {
      erc = "E2BIG - Argument list too long"s;
      break;
    }

    case ENOEXEC:
    {
      erc = "ENOEXEC - Exec format error"s;
      break;
    }

    case EBADF:
    {
      erc = "EBADF - Bad file number"s;
      break;
    }

    case ECHILD:
    {
      erc = "ECHILD - No child processes"s;
      break;
    }

    case EAGAIN:
    {
      erc = "EAGAIN - Try again"s;
      break;
    }

    case ENOMEM:
    {
      erc = "ENOMEM - Out of memory"s;
      break;
    }

    case EACCES:
    {
      erc = "EACCES - Permission denied"s;
      break;
    }

    case EFAULT:
    {
      erc = "EFAULT - Bad address"s;
      break;
    }

    case ENOTBLK:
    {
      erc = "ENOTBLK - Block device required"s;
      break;
    }

    case EBUSY:
    {
      erc = "EBUSY - Device or resource busy"s;
      break;
    }

    case EEXIST:
    {
      erc = "EEXIST - File exists"s;
      break;
    }

    case EXDEV:
    {
      erc = "EXDEV - Cross-device link"s;
      break;
    }

    case ENODEV:
    {
      erc = "ENODEV - No such device"s;
      break;
    }

    case ENOTDIR:
    {
      erc = "ENOTDIR - Not a directory"s;
      break;
    }

    case EISDIR:
    {
      erc = "EISDIR - Is a directory"s;
      break;
    }

    case EINVAL:
    {
      erc = "EINVAL - Invalid argument"s;
      break;
    }

    case ENFILE:
    {
      erc = "ENFILE - File table overflow"s;
      break;
    }

    case EMFILE:
    {
      erc = "EMFILE - Too many open files"s;
      break;
    }

    case ENOTTY:
    {
      erc = "ENOTTY - Not a typewriter"s;
      break;
    }

    case ETXTBSY:
    {
      erc = "ETXTBSY - Text file busy"s;
      break;
    }

    case EFBIG:
    {
      erc = "EFBIG - File too large"s;
      break;
    }

    case ENOSPC:
    {
      erc = "ENOSPC - No space left on device"s;
      break;
    }

    case ESPIPE:
    {
      erc = "ESPIPE - Illegal seek"s;
      break;
    }

    case EROFS:
    {
      erc = "EROFS - Read-only file system"s;
      break;
    }

    case EMLINK:
    {
      erc = "EMLINK - Too many links"s;
      break;
    }

    case EPIPE:
    {
      erc = "EPIPE - Broken pipe"s;
      break;
    }

    case EDOM:
    {
      erc = "EDOM - Math argument out of domain of func"s;
      break;
    }

    case ERANGE:
    {
      erc = "ERANGE - Math result not representable"s;
      break;
    }

    default:
    {
      ostringstream oss;
      oss << "Unknown pthread error: " << err;
      erc = oss.str();
      break;
    }
  }

  cerr << erc << endl;
   exit(1);
}

#endif
