//
// Created by raver119 on 22.11.2017.
//

#include <graph/FlatUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <array/ByteOrderUtils.h>


namespace nd4j {
    namespace graph {
        std::pair<int, int> FlatUtils::fromIntPair(IntPair *pair) {
            return std::pair<int, int>(pair->first(), pair->second());
        }

        std::pair<Nd4jIndex, Nd4jIndex> FlatUtils::fromLongPair(LongPair *pair) {
            return std::pair<Nd4jIndex, Nd4jIndex>(pair->first(), pair->second());
        }

        template<typename T>
        NDArray<T> *FlatUtils::fromFlatArray(const nd4j::graph::FlatArray *flatArray) {
            int * newShape = new int[shape::shapeInfoLength((int *)flatArray->shape()->data())];
            memcpy(newShape, flatArray->shape()->data(), shape::shapeInfoByteLength((int *)flatArray->shape()->data()));

            T * newBuffer = new T[shape::length(newShape)];
            DataTypeConversions<T>::convertType(newBuffer, (void *) flatArray->buffer()->data(), DataTypeUtils::fromFlatDataType(flatArray->dtype()), ByteOrderUtils::fromFlatByteOrder(flatArray->byteOrder()),  shape::length(newShape));
            auto array = new NDArray<T>(newBuffer, newShape);
            array->triggerAllocationFlag(true, true);

            return array;
        }


        template NDArray<float> *FlatUtils::fromFlatArray<float>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<float16> *FlatUtils::fromFlatArray<float16>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<double> *FlatUtils::fromFlatArray<double>(const nd4j::graph::FlatArray *flatArray);
    }
}