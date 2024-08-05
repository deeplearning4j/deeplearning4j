//
// Created by agibsonccc on 5/11/23.
//
#include <system/op_boilerplate.h>

#if defined(SD_GCC_FUNCTRACE)
void throwException(const char* exceptionMessage) {
#ifndef __CUDA_CC__
  StackTrace st;
  st.load_here(64);
  Printer p;
  p.print(st);
  throw std::runtime_error(exceptionMessage);
#else
   printf("Exception: %s\n", exceptionMessage);
#endif
}
#else
void throwException(const char* exceptionMessage) {
#ifndef __CUDA_CC__
  throw std::runtime_error(exceptionMessage);
#else
  printf("Exception: %s\n", exceptionMessage);
#endif
}
#endif