//
// Created by raver119 on 21.11.17.
//

#include <array/ByteOrderUtils.h>


namespace nd4j {
    ByteOrder ByteOrderUtils::fromFlatByteOrder(nd4j::graph::ByteOrder order) {
        return (ByteOrder) order;
    }
}