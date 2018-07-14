//
//  @author raver119@protonmail.com
//
#ifndef LIBND4J_U64_H
#define LIBND4J_U64_H

#include <cstdint>


namespace nd4j {
    typedef struct {
        int _v0;
        int _v1;
    } di32;

    typedef struct {
        uint32_t _v0;
        uint32_t _v1;
    } du32;

    typedef union
    {
        double _double;
        Nd4jLong _long;
        uint64_t _ulong;
        di32 _di32;
        du32 _du32;
    } u64;
}

#endif