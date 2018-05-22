//
// @author raver119@gmail.com
//

#include <helpers/ArrayUtils.h>

namespace nd4j {
    namespace ArrayUtils {
        void toIntPtr(std::initializer_list<int> list, int* target) {
            std::vector<int> vec(list);
            toIntPtr(vec, target);
        }

        void toIntPtr(std::vector<int>& list, int* target) {
            memcpy(target, list.data(), list.size() * sizeof(int));
        }

        void toLongPtr(std::initializer_list<Nd4jLong> list, Nd4jLong* target) {
            std::vector<Nd4jLong> vec(list);
            toLongPtr(vec, target);
        }

        void toLongPtr(std::vector<Nd4jLong>& list, Nd4jLong* target) {
            memcpy(target, list.data(), list.size() * sizeof(Nd4jLong));
        }

        std::vector<Nd4jLong> toLongVector(std::vector<int> vec) {
            std::vector<Nd4jLong> result(vec.size());

            for (Nd4jLong e = 0; e < vec.size(); e++)
                result[e] = vec[e];

            return result;
        }

        std::vector<Nd4jLong> toLongVector(std::vector<Nd4jLong> vec) {
            return vec;
        }
    }
}
