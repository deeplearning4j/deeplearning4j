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

    PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(zLen > Environment::getInstance()->elementwiseThreshold()) firstprivate(coords))
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
    PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    for(int i = 0; i < minDim; ++i)
        templatedSet<float>(buffer(), i*offset, this->dataType(), &v);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
    auto x = reinterpret_cast<T *>(xBuffer);
    auto y = reinterpret_cast<T *>(yBuffer);

    PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(schedule(static))
    for (Nd4jLong i = 0; i < length; ++i) {
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
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<Nd4jLong>& reps) const {
    const int repsSize = reps.size();

    Nd4jLong product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

    int rankOld = rankOf();
    int diff = rankOld - repsSize;
    if(product==1) {        // in this case 2 possibilities are present: just reshape or nothing to do
        NDArray result(*this);
        if(diff < 0) {      // reshape to higher dimension
            std::vector<Nd4jLong> shapeNew = reps;               // there is requirement to have unities at first "diff" positions of new shape
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
        for(Nd4jLong i=0;  i<resultLen; ++i) {
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

    auto repProd = shape::prodLong(reps.data(), reps.size());
    if (repProd < 1)
        throw std::runtime_error("NDArray::tile: reps can't contain 0s");

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
        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i*ews, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {

        for(Nd4jLong i=0;  i<targetLen; ++i) {

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

        for (Nd4jLong i = 0; i < targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else if(target.ordering() == 'c' && ews > 1) {

        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), i*ews, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {

        for(Nd4jLong i=0;  i<targetLen; ++i) {

            auto xOffset = target.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, target.getShapeInfo(), getShapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.getBuffer(), xOffset, getBuffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
NDArray* NDArray::repeat(int dimension, const std::vector<int>& repeats) const {
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

#include "NDArrayLambda.hpp"

#endif

/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}



#endif

