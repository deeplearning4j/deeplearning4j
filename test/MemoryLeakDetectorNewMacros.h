
/*
 * This file can be used to get extra debugging information about memory leaks in your production code.
 * It defines a preprocessor macro for operator new. This will pass additional information to the
 * operator new and this will give the line/file information of the memory leaks in your code.
 *
 * You can use this by including this file to all your production code. When using gcc, you can use
 * the -include file to do this for you.
 *
 * Warning: Using the new macro can cause a conflict with newly declared operator news. This can be
 * resolved by:
 * 1. #undef operator new before including this file
 * 2. Including the files that override operator new before this file.
 *    This can be done by creating your own NewMacros.h file that includes your operator new overrides
 *    and THEN this file.
 *
 * STL (or StdC++ lib) also does overrides for operator new. Therefore, you'd need to include the STL
 * files *before* this file too.
 *
 */

#include "CppUTestConfig.h"

/* Make sure that mem leak detection is on and that this is being included from a C++ file */
#if CPPUTEST_USE_MEM_LEAK_DETECTION && defined(__cplusplus)

/* This #ifndef prevents <new> from being included twice and enables the file to be included anywhere */
#ifndef CPPUTEST_USE_NEW_MACROS

    #if CPPUTEST_USE_STD_CPP_LIB
        #include <new>
        #include <memory>
        #include <string>
    #endif
#ifdef __CUDACC__
		__host__ __device__
#endif
    void* operator new(size_t size, const char* file, int line) UT_THROW (std::bad_alloc);
#ifdef __CUDACC__
		__host__ __device__
#endif    
void* operator new[](size_t size, const char* file, int line) UT_THROW (std::bad_alloc);
#ifdef __CUDACC__
		__host__ __device__
#endif    
void* operator new(size_t size) UT_THROW(std::bad_alloc);
#ifdef __CUDACC__
		__host__ __device__
#endif    
void* operator new[](size_t size) UT_THROW(std::bad_alloc);
#ifdef __CUDACC__
		__host__ __device__
#endif
    void operator delete(void* mem) UT_NOTHROW;
#ifdef __CUDACC__
		__host__ __device__
#endif    
void operator delete[](void* mem) UT_NOTHROW;
#ifdef __CUDACC__
		__host__ __device__
#endif    
void operator delete(void* mem, const char* file, int line) UT_NOTHROW;
    void operator delete[](void* mem, const char* file, int line) UT_NOTHROW;

#endif


#ifdef __clang__
 #pragma clang diagnostic push
 #if __clang_major__ >= 3 && __clang_minor__ >= 6
  #pragma clang diagnostic ignored "-Wkeyword-macro"
 #endif
#endif

#define new new(__FILE__, __LINE__)

#ifdef __clang__
 #pragma clang diagnostic pop
#endif

#define CPPUTEST_USE_NEW_MACROS 1

#endif
