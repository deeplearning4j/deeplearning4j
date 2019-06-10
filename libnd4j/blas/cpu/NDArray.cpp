/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NDArrayFactory.h"
#include "NativeOpExecutioner.h"
#include <BroadcastPairwiseConverter.h>
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform_same.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>
#include <MmulHelper.h>
#include <helpers/threshold.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/allocation_exception.h>
#include <helpers/ConstantTadHelper.h>

#include <NDArray.hpp>


namespace nd4j {

////////////////////////////////////////////////////////////////////////

void* NDArray::platformBuffer()             { return buffer();    }
void* NDArray::getPlatformBuffer() const    { return getBuffer(); }

Nd4jLong* NDArray::getPlatformShapeInfo() const { return getShapeInfo(); }
Nd4jLong* NDArray::platformShapeInfo()          { return shapeInfo(); }

void NDArray::syncToDevice() const          { }
void NDArray::syncToHost() const            { }
void NDArray::tickWriteHost() const         { }
void NDArray::tickWriteDevice() const       { }
void NDArray::tickReadHost() const          { }
void NDArray::tickReadDevice() const        { }
void NDArray::tickBothActual() const        { }
bool NDArray::isActualOnHostSide() const    { return true; }
bool NDArray::isActualOnDeviceSide() const  { return true; }
void NDArray::makeBothBuffersActual() const { }


////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::fillAsTriangular(const float val, int lower, int upper, const char direction, NDArray* target) {

    if (isS())
        throw std::runtime_error("NDArray::fillArrayAsTriangular: you can't use this method on String array!");

    if(target == nullptr)
        target = this;

    if(!isSameShape(target) && !(rankOf() == 1 && target->rankOf() == 2 && sizeAt(0) == target->sizeAt(0) && sizeAt(0) == target->sizeAt(1)))
        throw std::string("NDArray::fillArrayAsTriangular method: wrong shape of target array !");

    if (direction == 'u')
        lower = -target->sizeAt(-2);
    else if (direction == 'l')
        upper = target->sizeAt(-1);

    const T value = static_cast<T>(val);
    const auto x = reinterpret_cast<const T*>(getBuffer());
          auto z = reinterpret_cast<T*>(target->getBuffer());

    const int xRank = rankOf();
    const int zRank = target->rankOf();

    const auto zLen = target->lengthOf();

    const bool areSameOffsets = shape::haveSameShapeAndStrides(getShapeInfo(), target->getShapeInfo());

    std::vector<Nd4jLong> coords(zRank);

    PRAGMA_OMP_PARALLEL_FOR_ARGS(if(zLen > Environment::getInstance()->elementwiseThreshold()) firstprivate(coords))
    for (Nd4jLong i = 0; i < zLen; ++i) {

        shape::index2coords(zRank, target->shapeOf(), i, zLen, coords.data());
        const auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), coords.data(), zRank);

        // if( (row + upper < col) || (row + lower > col) )
        if((coords[zRank - 2] + upper < coords[zRank - 1]) || (coords[zRank - 2] + lower > coords[zRank - 1]))
            z[zOffset] = value;
        else if(this != target) {      // when this and target are different arrays
            if(xRank != zRank)
                coords[0] = coords[1];
            const auto xOffset = areSameOffsets ? zOffset : shape::getOffset(0, shapeOf(), stridesOf(), coords.data(), xRank);
            z[zOffset] = x[xOffset];
        }
    }
}
BUILD_SINGLE_TEMPLATE(template void NDArray::fillAsTriangular, (const float val, int lower, int upper, const char direction, NDArray* target), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
    if (isS())
        throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

    this->nullify();

    int  rank    = rankOf();
    auto shape   = shapeOf();
    auto strides = stridesOf();
    int  minDim  = MAX_INT;
    Nd4jLong indices[MAX_RANK];
    for(int j = 0; j < rank; ++j)
        indices[j] = 1;

    Nd4jLong offset = shape::getOffset(0, shape, strides, indices, rank);

    for(int i = 0; i < rank; ++i)
        if(minDim > shape[i])
            minDim = shape[i];

    float v = 1.0f;
    PRAGMA_OMP_PARALLEL_FOR_ARGS(if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    for(int i = 0; i < minDim; ++i)
        templatedSet<float>(buffer(), i*offset, this->dataType(), &v);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
    auto x = reinterpret_cast<T *>(xBuffer);
    auto y = reinterpret_cast<T *>(yBuffer);

    PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(schedule(static))
    for (int i = 0; i < length; ++i) {
        auto temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}
BUILD_SINGLE_TEMPLATE(template void templatedSwap, (void *xBuffer, void *yBuffer, Nd4jLong length), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::swapUnsafe(NDArray& other) {
    auto xType = this->dataType();

    if (xType != other.dataType())
        throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

    if(buffer() == nullptr || other.buffer() == nullptr)
        throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

    if(lengthOf() != other.lengthOf())
        throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

    BUILD_SINGLE_SELECTOR(xType, templatedSwap, (buffer(), other.buffer(), this->lengthOf()), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
    // no-op
}
void NDArray::prepareSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {
    // no-op
}
void NDArray::registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {
    // no-op
}
void NDArray::preparePrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {
    // no-op
}
void NDArray::registerPrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {
    // no-op
}
void NDArray::syncShape() const {
    // no-op
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) const {

}

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
    return nullptr;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<Nd4jLong>& reps) const {
    int dim = reps.size();
    int product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

    int rankOld = rankOf();
    int diff = rankOld - dim;
    if(product==1) {        // in this case 2 possibilities are present: just reshape or nothing to do
        NDArray result(*this);
        if(diff < 0) {      // reshape to higher dimension
            std::vector<Nd4jLong> shapeNew = reps;               // need to have unities at first "diff" positions of new shape
            memcpy(&shapeNew[-diff], result.getShapeInfo()+1, rankOld * sizeof(Nd4jLong));   // put old shape numbers at rest of positions
            result.reshapei(ordering(), shapeNew);
        }
        return result;             // nothing to do, if diff >= 0 -> identity tile
    }

    // evaluate shapeInfo for resulting array
    auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
    // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
    std::shared_ptr<DataBuffer> newBuff = std::make_shared<DataBuffer>(shape::length(newShapeInfo) * sizeOfT(), dataType(), getContext()->getWorkspace());
    // assign new shape and new buffer to resulting array
    NDArray result(newBuff, ShapeDescriptor(newShapeInfo), getContext());

    // fill newBuff, loop through all elements of newBuff
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const auto resultLen = result.lengthOf();
    auto xType = this->dataType();
    if(result.ordering() == 'c') {           //  ews == 1 always here

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i = 0;  i < resultLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, newShapeInfo, getShapeInfo());
            BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (result.getBuffer(), i, this->getBuffer(), yOffset), LIBND4J_TYPES);

        }
    }
    else {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(int i=0;  i<resultLen; ++i) {
            auto xOffset = result.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, newShapeInfo, getShapeInfo());
            BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (result.getBuffer(), xOffset, this->getBuffer(), yOffset), LIBND4J_TYPES);
        }
    }
    result.tickWriteHost();
    return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tile(const std::vector<Nd4jLong>& reps, NDArray& target) const {

    // evaluate true tile shapeInfo for comparison with target shapeInfo
    auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
    if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
        delete []newShapeInfo;
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
    }

    // fill newBuff, loop through all elements of newBuff
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const int targetLen = target.lengthOf();
    if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else if(target.ordering() == 'c' && ews > 1) {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i*ews, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i) {

            auto xOffset = target.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), xOffset, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) const {
    if(rankOf() > target.rankOf())
        throw std::runtime_error("NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

    if(!ShapeUtils::areShapesBroadcastable(*this, target))
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

    // fill newBuff, loop through all elements of newBuff
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const auto ews = target.ews();
    const auto targetLen = target.lengthOf();
    if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for (int i = 0; i < targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else if(target.ordering() == 'c' && ews > 1) {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i*ews, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {

//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i) {

            auto xOffset = target.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), xOffset, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
NDArray* NDArray::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {
    auto outShape = ShapeUtils::evalRepeatShape(dimension, repeats, *this);

    // the size of outShape == rank
    int rank = rankOf();            // = outShape.size()

    auto ret = new NDArray('c', outShape, dataType(),  getContext());

    auto retArrs  = ret->allTensorsAlongDimension({dimension});
    auto thisArrs = this->allTensorsAlongDimension({dimension});

    auto repeatDelta = shape::prodLong(outShape.data(), rank) / this->lengthOf();
    auto numTads = retArrs->size();

    for (int i = 0; i < numTads; i++) {
        auto thisTensor = thisArrs->at(i);
        auto retTensor  = retArrs->at(i);
        Nd4jLong retIdx = 0;

        for (Nd4jLong k = 0; k < thisTensor->lengthOf(); k++) {
            auto s = thisTensor->e(k);
            for (Nd4jLong j = 0; j < repeatDelta; j++)
                retTensor->p(retIdx++, s);
        }
    }

    delete retArrs;
    delete thisArrs;
    return ret;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
void NDArray::repeat(int dimension, NDArray& target) const {

    if(dimension < 0)
        dimension += rankOf();

    if(rankOf() != target.rankOf())
        throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong rank of target array it must be equal to this array rank!");

    Nd4jLong repeatDelta = target.sizeAt(dimension) / sizeAt(dimension);

    if(repeatDelta == 0)
        throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong shape of target array!");


    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
    const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(getShapeInfo(), dimsToExclude);

    for (int i = 0; i < numTads; i++) {
        auto thisTensor = (*this)(i, dimsToExclude);
        auto retTensor = target(i, dimsToExclude);
        int tensorLength = thisTensor.lengthOf();
        int retIdx = 0;
        if (isR()) {
            for (int k = 0; k < tensorLength; k++) {
                auto s = thisTensor.e<double>(k);
                for (int j = 0; j < repeatDelta; j++) {
                    retTensor.p<double>(retIdx++, s);
                }
            }
        } else {
            for (int k = 0; k < tensorLength; k++) {
                auto s = thisTensor.e<Nd4jLong>(k);
                for (int j = 0; j < repeatDelta; j++) {
                    retTensor.p<Nd4jLong>(retIdx++, s);
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__

template<typename T>
void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<T(T, T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (second == nullptr) {
        nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Second is NULL\n","");
        throw std::runtime_error("second is null");
    }

    if (third == nullptr) {
        nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Third is NULL\n","");
        throw std::runtime_error("third is null");
    }
    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != second->dataType() || dataType() != third->dataType() || dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: bother four arrays (this, second, third, target) should have the same type !");

    if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) || !this->isSameShape(third)) {
        nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = second->bufferAsT<T>();
    auto t = third->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == second->ordering() && this->ordering() == third->ordering()  && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == second->ews() && this->ews() == third->ews()) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong e = 0; e < _length; e++)
            z[e] = func(f[e], s[e], t[e]);
    } else {
        if (f == z) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto tOffset = this->getOffset(e);
                auto uOffset = second->getOffset(e);
                auto vOffset = third->getOffset(e);

                f[tOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto tOffset = this->getOffset(e);
                auto uOffset = second->getOffset(e);
                auto vOffset = third->getOffset(e);
                auto zOffset = target->getOffset(e);

                z[zOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
            }
        }
    }
}
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<double (double, double, double)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float (float, float, float)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float16 (float16, float16, float16)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bfloat16 (bfloat16, bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int (int, int, int)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int16_t (int16_t, int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint8_t (uint8_t, uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint16_t (uint16_t, uint16_t, uint16_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint32_t (uint32_t, uint32_t, uint32_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint64_t (uint64_t, uint64_t, uint64_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int8_t (int8_t, int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bool (bool, bool, bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<T(T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (other == nullptr) {
        nd4j_printf("applyPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
        throw std::runtime_error("Other is null");
    }

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != other->dataType() || dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: all three arrays (this, other, target) must have the same type !");

    if (this->lengthOf() != other->lengthOf()) {
        nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = other->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < _length; e++)
            z[e] = func(f[e], s[e]);
    } else {
        if (f == z) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto yOffset = other->getOffset(e);

                f[xOffset] = func(f[xOffset], s[yOffset]);
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto yOffset = other->getOffset(e);
                auto zOffset = target->getOffset(e);

                z[zOffset] = func(f[xOffset], s[yOffset]);
            }
        }
    }
}
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<double (double, double)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float (float, float)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float16 (float16, float16)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bfloat16 (bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int (int, int)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int16_t (int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint8_t (uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint16_t (uint16_t, uint16_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint32_t (uint32_t, uint32_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint64_t (uint64_t, uint64_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int8_t (int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bool (bool, bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyLambda(const std::function<T(T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyLambda<T> method: types of this and target array should match !");

    auto f = this->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < _length; e++)
            z[e] = func(f[e]);
    } else {
        if (f == z) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);

                f[xOffset] = func(f[xOffset]);
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto zOffset = target->getOffset(e);

                z[zOffset] = func(f[xOffset]);
            }
        }
    }
}
template void NDArray::applyLambda(const std::function<double(double)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<float(float)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<float16(float16)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<bfloat16(bfloat16)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<Nd4jLong(Nd4jLong)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<int16_t(int16_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<int32_t(int32_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint8_t(uint8_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint16_t(uint16_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<uint32_t(uint32_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<uint64_t(uint64_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int8_t(int8_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<bool(bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyIndexedLambda<T> method: types of this and target array should match !");

    auto f = this->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong e = 0; e < _length; e++)
            z[e] = func(e, f[e]);
    } else {
        if (f == z) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (Nd4jLong e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);

                f[xOffset] = func(e, f[xOffset]);
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (Nd4jLong e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto zOffset = target->getOffset(e);

                z[zOffset] = func(e, f[xOffset]);
            }
        }
    }
}
template void NDArray::applyIndexedLambda(const std::function<double(Nd4jLong, double)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<float(Nd4jLong, float)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<float16(Nd4jLong, float16)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<bfloat16(Nd4jLong, bfloat16)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<Nd4jLong(Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<int(Nd4jLong, int)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<int16_t(Nd4jLong, int16_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint8_t (Nd4jLong, uint8_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint16_t (Nd4jLong, uint16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<uint32_t (Nd4jLong, uint32_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<uint64_t (Nd4jLong, uint64_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int8_t(Nd4jLong, int8_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<bool(Nd4jLong, bool)>& func, NDArray* target);

////////////////////////////////////////////////////////////////////////
    void* NDArray::specialBuffer() {

        if (_buffer->special() == nullptr)
            return getBuffer();
        // FIXME: this should be fixed once CUDA backend added
        return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
    }

////////////////////////////////////////////////////////////////////////
    void* NDArray::getSpecialBuffer() const {
        if (_buffer->special() == nullptr)
            return getBuffer();
        // FIXME: this should be fixed once CUDA backend added
        return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
    }

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (other == nullptr) {
        nd4j_printf("applyIndexedPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
        throw std::runtime_error("Other is null");
    }
    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: types of this and target array should match !");
    if (this->lengthOf() != other->lengthOf()) {
        nd4j_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = other->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong e = 0; e < _length; e++)
            z[e] = func((Nd4jLong) e, f[e], s[e]);
    } else {
        if (f == z) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto yOffset = other->getOffset(e);

                f[xOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int e = 0; e < _length; e++) {

                auto xOffset = this->getOffset(e);
                auto yOffset = other->getOffset(e);
                auto zOffset = target->getOffset(e);

                z[zOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
            }
        }
    }
}
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<double (Nd4jLong, double, double)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float (Nd4jLong, float, float)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float16 (Nd4jLong, float16, float16)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bfloat16 (Nd4jLong, bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int (Nd4jLong, int, int)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int16_t (Nd4jLong, int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint8_t (Nd4jLong, uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint16_t (Nd4jLong, uint16_t, uint16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint32_t (Nd4jLong, uint32_t, uint32_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint64_t (Nd4jLong, uint64_t, uint64_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int8_t (Nd4jLong, int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bool (Nd4jLong, bool, bool)>& func, NDArray* target);
#endif

/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}



#endif

