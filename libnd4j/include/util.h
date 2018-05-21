/* 
 * File:   util.h
 * Author: saudet
 *
 * Created on July 18, 2016, 1:28 PM
 */

#ifndef NATIVEOPERATIONS_UTIL_H
#define NATIVEOPERATIONS_UTIL_H

#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "pointercast.h"

static inline Nd4jLong microTime() {
#ifdef WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (Nd4jLong)count.QuadPart/freq.QuadPart;
#else
    timeval tv;
    gettimeofday(&tv, NULL);
    return (Nd4jLong)tv.tv_sec*1000000 + tv.tv_usec;
#endif
}

#endif /* NATIVEOPERATIONS_UTIL_H */
