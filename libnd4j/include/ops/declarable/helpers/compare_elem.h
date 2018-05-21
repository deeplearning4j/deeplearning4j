
#ifndef LIBND4J_COMPARE_ELEM_H
#define LIBND4J_COMPARE_ELEM_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void compare_elem(NDArray<T>* input, bool isStrictlyIncreasing, bool& output);
        }
    }
}


#endif //LIBND4J_COMPARE_ELEM_H