//
// @author raver119@gmail.com
//

#ifndef LIBND4J_ARRAYUTILS_H
#define LIBND4J_ARRAYUTILS_H

#include <initializer_list>
#include <vector>
#include <cstring>

namespace nd4j {
    namespace ArrayUtils {


        void toIntPtr(std::initializer_list<int> list, int* target);
        void toIntPtr(std::vector<int>& list, int* target);
    }
}

void nd4j::ArrayUtils::toIntPtr(std::initializer_list<int> list, int* target) {
    std::vector<int> vec(list);
    toIntPtr(vec, target);
}

void nd4j::ArrayUtils::toIntPtr(std::vector<int>& list, int* target) {
    memcpy(target, list.data(), list.size() * sizeof(int));
}

#endif //LIBND4J_ARRAYUTILS_H
