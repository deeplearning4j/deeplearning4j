//
// Created by raver119 on 22.11.2017.
//

#ifndef LIBND4J_FLATUTILS_H
#define LIBND4J_FLATUTILS_H

#include <utility>
#include <pointercast.h>
#include <graph/generated/array_generated.h>
#include <graph/generated/node_generated.h>
#include <NDArray.h>

namespace nd4j {
    namespace graph {
        class FlatUtils {
        public:
            static std::pair<int, int> fromIntPair(IntPair* pair);

            static std::pair<Nd4jLong, Nd4jLong > fromLongPair(LongPair* pair);

            template <typename T>
            static NDArray<T>* fromFlatArray(const nd4j::graph::FlatArray* flatArray);
        };
    }
}

#endif //LIBND4J_FLATUTILS_H
