//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDARRAYFACTORY_H
#define LIBND4J_NDARRAYFACTORY_H

#include "NDArray.h"
#include <array/ResultSet.h>

namespace nd4j {
    template<typename T>
    class NDArrayFactory {
   

    public:
        static NDArray<T>* createUninitialized(NDArray<T>* other);

        static ResultSet<T>* multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions);

        static ResultSet<T>* allTensorsAlongDimension(const NDArray<T>* ndArray, const std::vector<int> &dimensions);

        static ResultSet<T>* allExamples(NDArray<T>* ndArray);

        static ResultSet<T>* allTensorsAlongDimension(const NDArray<T>* ndArray, const std::initializer_list<int> dimensions);

        static NDArray<T>* tile(NDArray<T> *original, std::vector<int>& dimensions);

        static NDArray<T>* repeat(NDArray<T> *original, std::vector<int>& repeats);

        static NDArray<T>* scalar(T value);

        static NDArray<T>* valueOf(std::initializer_list<Nd4jLong> shape, T value, char order = 'c');
        static NDArray<T>* valueOf(std::vector<Nd4jLong>& shape, T value, char order = 'c');

    };
}


#endif //LIBND4J_NDARRAYFACTORY_H
