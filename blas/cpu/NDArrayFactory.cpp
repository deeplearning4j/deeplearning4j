//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_NDARRAYFACTORY_CPP
#define LIBND4J_NDARRAYFACTORY_CPP

#include "../NDArrayFactory.h"
#include "../NDArray.h"
#include "NDArray.cpp"

namespace nd4j {
    template<typename T>
    ArrayList<T>* NDArrayFactory::multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions) {
        auto result = new ArrayList<T>();

        if (indices.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(ndArray->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTads = ndArray->lengthOf() / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(ndArray->_shapeInfo, copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        int* shapeInfo = new int[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("Index %i is higher then number of TADs: %i\n", idx, numTads);
                throw "Bad index";
            }


            T* buffer = ndArray->_buffer + tad->tadOffsets[idx];
            auto array = new NDArray<T>(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->_isShapeAlloc = true;
        else
            delete[] shapeInfo;

        return result;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory::tile(NDArray<T> *original, std::vector<int> &dimensions) {
        return nullptr;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory::repeat(NDArray<T> *original, std::vector<int> &repeats) {
        return nullptr;
    }
}

#endif