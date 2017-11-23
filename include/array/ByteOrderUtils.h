//
// Created by raver119 on 21.11.17.
//

#ifndef LIBND4J_BYTEORDERUTILS_H
#define LIBND4J_BYTEORDERUTILS_H

#include <graph/generated/array_generated.h>
#include "ByteOrder.h"

namespace nd4j {
    class ByteOrderUtils {
    public:
        static ByteOrder fromFlatByteOrder(nd4j::graph::ByteOrder order);
    };
}


#endif //LIBND4J_BYTEORDERUTILS_H
