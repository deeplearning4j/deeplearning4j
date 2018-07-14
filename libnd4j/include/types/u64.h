//
//  @author raver119@protonmail.com
//
#ifndef LIBND4J_U64_H
#define LIBND4J_U64_H

#include <cstdint>


namespace nd4j {
    typedef struct {
        int _i0;
        int _i1;
    } di;

    typedef union
    {
        double _double;
        Nd4jLong _long;
        di _di;
    } u64;
}

#endif