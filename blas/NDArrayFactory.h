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
        static ArrayList<T>* multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions);


        template<typename T>
        static NDArray<T>* tile(NDArray<T> *original, std::vector<int>& dimensions);


        template<typename T>
        static NDArray<T>* repeat(NDArray<T> *original, std::vector<int>& repeats);
    };
}


#endif //LIBND4J_NDARRAYFACTORY_H
