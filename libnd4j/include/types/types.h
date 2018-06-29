//
// Created by raver on 6/12/2018.
//

#ifndef LIBND4J_TYPES_H
#define LIBND4J_TYPES_H

#include <pointercast.h>

//// Forward declarations of custom types
struct float16;
namespace nd4j {
    struct float8;
    struct int8;
    struct uint8;
    struct int16;
    struct uint16;
}

#define LIBND4J_TYPES \
        float, \
        float16, \
        nd4j::float8, \
        double, \
        int, \
        Nd4jLong, \
        nd4j::int8, \
        nd4j::uint8, \
        nd4j::int16, \
        nd4j::uint16

#endif //LIBND4J_TYPES_H
