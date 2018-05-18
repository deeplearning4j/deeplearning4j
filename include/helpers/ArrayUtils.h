//
// @author raver119@gmail.com
//

#ifndef LIBND4J_ARRAYUTILS_H
#define LIBND4J_ARRAYUTILS_H

#include <initializer_list>
#include <vector>
#include <cstring>
#include <pointercast.h>

namespace nd4j {
    namespace ArrayUtils {
        void toIntPtr(std::initializer_list<int> list, int* target);
        void toIntPtr(std::vector<int>& list, int* target);

        void toLongPtr(std::initializer_list<Nd4jLong> list, Nd4jLong* target);
        void toLongPtr(std::vector<Nd4jLong>& list, Nd4jLong* target);


        std::vector<Nd4jLong> toLongVector(std::vector<int> vec);
        std::vector<Nd4jLong> toLongVector(std::vector<Nd4jLong> vec);
    }
}

#endif //LIBND4J_ARRAYUTILS_H
