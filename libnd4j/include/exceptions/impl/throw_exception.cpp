//
// Created by agibsonccc on 5/11/23.
//
#include <system/op_boilerplate.h>
#include <execution/LaunchContext.h>

#if defined(SD_GCC_FUNCTRACE)
void throwException(const char* exceptionMessage) {
#ifndef __CUDA_CC__
  StackTrace st;
  st.load_here(64);
  Printer p;
  p.print(st);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(exceptionMessage);

  throw std::runtime_error(exceptionMessage);
#else
  LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  printf("Exception: %s\n", exceptionMessage);
#endif
}
#else
void throwException(const char* exceptionMessage) {
#ifndef __CUDA_CC__
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  throw std::runtime_error(exceptionMessage);
#else
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  printf("Exception: %s\n", exceptionMessage);
#endif
}
#endif