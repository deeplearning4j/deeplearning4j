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

    private:
        // helpers for helper 
        // multiptication N-dimensions tensor on other N-dimensions one
        static nd4j::NDArray<T>* mmulHelperNxN(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);
        // multiptication Matrix to vector
        static nd4j::NDArray<T>* mmulHelperMxV(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);
        // multiptication Matrix to Matrix
        static nd4j::NDArray<T>* mmulHelperMxM(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);



    public:
        static NDArray<T>* createUninitialized(NDArray<T>* other);

        static ResultSet<T>* multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions);

        static ResultSet<T>* allTensorsAlongDimension(const NDArray<T>* ndArray, const std::vector<int> &dimensions);

        static ResultSet<T>* allExamples(NDArray<T>* ndArray);

        static ResultSet<T>* allTensorsAlongDimension(const NDArray<T>* ndArray, const std::initializer_list<int> dimensions);

        static NDArray<T>* tile(NDArray<T> *original, std::vector<int>& dimensions);

        static NDArray<T>* repeat(NDArray<T> *original, std::vector<int>& repeats);

        static nd4j::NDArray<T>* mmulHelper(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C = nullptr, T alpha = 1.0f, T beta = 0.0f);

        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB = {});

        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::vector<int>& axesA, const std::vector<int>& axesB);

        static void tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC = {});

#ifndef __JAVACPP_HACK__
        /**
        *  modif - (can be empty) vector containing a subsequence of permutation/reshaping arrays (in any order), user must take care of correctness of such arrays by himself 
        */
        static void tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC);
        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB);
#endif

        static NDArray<T>* linspace(T from, T to, Nd4jLong numElements);
        
        static void linspace(T from, NDArray<T>& arr, T step = 1.0f);

        static NDArray<T>* scalar(T value);

        static NDArray<T>* valueOf(std::initializer_list<Nd4jLong> shape, T value, char order = 'c');
        static NDArray<T>* valueOf(std::vector<Nd4jLong>& shape, T value, char order = 'c');

        static NDArray<T>* concat(const std::vector<NDArray<T> *>& vectors, int axis = 0, NDArray<T>* target = nullptr);

        static NDArray<T>* simpleMMul(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c , const T alpha, const T beta);

        static void matmul(const nd4j::NDArray<T>* x, const nd4j::NDArray<T>* y, nd4j::NDArray<T>* z, const bool transX, const bool transY);
    };
}


#endif //LIBND4J_NDARRAYFACTORY_H
