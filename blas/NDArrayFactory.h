//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDARRAYFACTORY_H
#define LIBND4J_NDARRAYFACTORY_H

#include "NDArray.h"
#include <graph/ArrayList.h>

namespace nd4j {
    class NDArrayFactory {
    public:

        template<typename T>
        static NDArray<T>* createUninitialized(NDArray<T>* other);

        template<typename T>
        static ArrayList<T>* multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions);

        template<typename T>
        static ArrayList<T>* allTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &dimensions);

        template<typename T>
        static ArrayList<T>* allExamples(NDArray<T>* ndArray);

        template<typename T>
        static ArrayList<T>* allTensorsAlongDimension(NDArray<T>* ndArray, std::initializer_list<int> dimensions);

        template<typename T>
        static NDArray<T>* tile(NDArray<T> *original, std::vector<int>& dimensions);


        template<typename T>
        static NDArray<T>* repeat(NDArray<T> *original, std::vector<int>& repeats);

        template<typename T>
        static nd4j::NDArray<T>* mmulHelper(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C = nullptr, T alpha = 1.0f, T beta = 0.0f);

        template<typename T>
        static NDArray<T>* linspace(T from, T to, Nd4jIndex numElements);


    };
}


#endif //LIBND4J_NDARRAYFACTORY_H
