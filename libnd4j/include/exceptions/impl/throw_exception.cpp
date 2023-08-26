//
// Created by agibsonccc on 5/11/23.
//
#include <system/op_boilerplate.h>

#if defined(SD_GCC_FUNCTRACE)
void throwException(const char* exceptionMessage) {
  StackTrace st;
  st.load_here(64);
  Printer p;
  p.print(st);
  throw std::runtime_error(exceptionMessage);
}
#else
void throwException(const char* exceptionMessage) {
  throw std::runtime_error(exceptionMessage);
}
#endif