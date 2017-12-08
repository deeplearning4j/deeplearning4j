//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PREFIX_HELPER_H
#define LIBND4J_PREFIX_HELPER_H

#include <pointercast.h>
#include <types/float16.h>
#include <vector>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T, typename OpName>
            void _prefix(T* x, int* xShapeInfo, T* z, int* zShapeInfo);

            template <typename T, typename OpName>
            void _prefix(NDArray<T>* x, NDArray<T>* z, std::vector<int>& dims);
        }
    }
}

#endif