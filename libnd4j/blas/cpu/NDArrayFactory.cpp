//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_NDARRAYFACTORY_CPP
#define LIBND4J_NDARRAYFACTORY_CPP

#include "../NDArrayFactory.h"
#include "../NDArray.h"
#include <memory/Workspace.h>
#include <ops/gemm.h>
#include <types/float16.h>
#include <helpers/ShapeUtils.h>
#include <helpers/BlasHelper.h>

namespace nd4j {

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allExamples(NDArray<T>* ndArray) {
        
        std::vector<int> dimensions(ndArray->rankOf() - 1);            
        for (int e = 1; e < ndArray->rankOf(); e++)
            dimensions[e-1] = e;

        return allTensorsAlongDimension(ndArray, dimensions);
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions) {
        auto result = new ResultSet<T>();

        if (indices.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(ndArray->getShapeInfo(), copy.data(), copy.size());
        Nd4jLong numTads = ndArray->lengthOf() / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(ndArray->getShapeInfo(), copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        // FIXME: why we're not using workspaces here?
        auto shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("Index %i is higher then number of TADs: %i\n", idx, numTads);
                throw std::runtime_error("Bad index");
            }


            T* buffer = ndArray->getBuffer() + tad->tadOffsets[idx];
            auto array = new NDArray<T>(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allTensorsAlongDimension(const NDArray<T>* ndArray, const std::initializer_list<int> dimensions) {
        std::vector<int> vec(dimensions);
        return allTensorsAlongDimension(ndArray, vec);
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allTensorsAlongDimension(const NDArray<T>* ndArray, const std::vector<int> &dimensions) {
        auto result = new ResultSet<T>();

        if(dimensions.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        if(copy.back() >= ndArray->rankOf())
            throw std::runtime_error("NDArrayFactory::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");

        Nd4jLong tadLength = shape::tadLength(ndArray->getShapeInfo(), copy.data(), copy.size());
        Nd4jLong numTads = ndArray->lengthOf() / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(ndArray->getShapeInfo(), copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        auto shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (int idx = 0; idx < numTads; idx++ ) {
            T* buffer = const_cast<NDArray<T>*>(ndArray)->getBuffer() + tad->tadOffsets[idx];
            auto array = new NDArray<T>(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

     //////////////////////////////////////////////////////////////////////////////

    template<typename T>
    NDArray<T>* NDArrayFactory<T>::tile(NDArray<T> *original, std::vector<int> &dimensions) {
        return nullptr;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory<T>::repeat(NDArray<T> *original, std::vector<int> &repeats) {
        return nullptr;
    }

    

    template<typename T>
    NDArray<T>* NDArrayFactory<T>::createUninitialized(NDArray<T>* other) {
        auto workspace = other->getWorkspace();

        Nd4jLong* newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(other->getShapeInfo()), Nd4jLong);
        memcpy(newShape, other->getShapeInfo(), shape::shapeInfoByteLength(other->getShapeInfo()));

        T* buffer;
        ALLOCATE(buffer, workspace, other->lengthOf(), T);
        auto result = new NDArray<T>(buffer, newShape, workspace);
        result->triggerAllocationFlag(true, true);

        return result;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::scalar(T value) {
        auto res = new NDArray<T>('c', {1, 1});
        res->putScalar(0, value);

        return res;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::valueOf(std::initializer_list<Nd4jLong> shape, T value, char order) {
        auto result = new NDArray<T>(order, shape);
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::valueOf(std::vector<Nd4jLong>& shape, T value, char order) {
        auto result = new NDArray<T>(order, shape);
        result->assign(value);
        return result;
    }

template class ND4J_EXPORT NDArrayFactory<float>;
template class ND4J_EXPORT NDArrayFactory<float16>;
template class ND4J_EXPORT NDArrayFactory<double>;
}


#endif
