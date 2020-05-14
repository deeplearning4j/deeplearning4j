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

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/BroadcastPairwiseConverter.h>
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops/ops.h>
#include <ops/gemm.h>
#include <system/pointercast.h>
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
#include <helpers/MmulHelper.h>
#include <helpers/threshold.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/allocation_exception.h>
#include <helpers/ConstantTadHelper.h>

#include <array/NDArray.hXX>


namespace sd {

////////////////////////////////////////////////////////////////////////

void* NDArray::platformBuffer()             { return buffer();    }
void const* NDArray::platformBuffer() const    { return buffer(); }

Nd4jLong const* NDArray::platformShapeInfo() const { return shapeInfo(); }

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
void NDArray::fillAsTriangular(const float val, int lower, int upper, NDArray& target, const char direction) {

    if (isS())
        throw std::runtime_error("NDArray::fillArrayAsTriangular: you can't use this method on String array!");

    if(!isSameShape(target) && !(rankOf() == 1 && target.rankOf() == 2 && sizeAt(0) == target.sizeAt(0) && sizeAt(0) == target.sizeAt(1)))
        throw std::string("NDArray::fillArrayAsTriangular method: wrong shape of target array !");

    if (direction == 'u')
        lower = -target.sizeAt(-2);
    else if (direction == 'l')
        upper = target.sizeAt(-1);

    const T value = static_cast<T>(val);
    const auto x = reinterpret_cast<const T*>(buffer());
          auto z = reinterpret_cast<T*>(target.buffer());

    const int xRank = rankOf();
    const int zRank = target.rankOf();

    const auto zLen = target.lengthOf();

    const bool areSameOffsets = shape::haveSameShapeAndStrides(shapeInfo(), target.shapeInfo());

    auto func = PRAGMA_THREADS_FOR {

        int coords[MAX_RANK], temp;

        for (auto i = start; i < stop; i++) {

            shape::index2coordsCPU(start, i, target.shapeInfo(), coords);
            const auto zOffset = shape::getOffset(target.shapeInfo(), coords);

            // if( (row + upper < col) || (row + lower > col) )
            if ((coords[zRank - 2] + upper < coords[zRank - 1]) || (coords[zRank - 2] + lower > coords[zRank - 1]))
                z[zOffset] = value;
            else if (this != &target) {      // when this and target are different arrays
                if (xRank != zRank) {
                    temp = coords[0];
                    coords[0] = coords[1];
                }

                const auto xOffset = areSameOffsets ? zOffset : shape::getOffset(shapeInfo(), coords);
                z[zOffset] = x[xOffset];

                if (xRank != zRank)     // restore first coordinate
                    coords[0] = temp;
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, zLen);
}
BUILD_SINGLE_TEMPLATE(template void NDArray::fillAsTriangular, (const float val, int lower, int upper, NDArray& target, const char direction), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
    if (isS())
        throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

    this->nullify();

    int  rank    = rankOf();
    auto shape   = shapeOf();
    int  minDim  = MAX_INT;
    Nd4jLong indices[MAX_RANK];
    for(int j = 0; j < rank; ++j)
        indices[j] = 1;

    Nd4jLong offset = shape::getOffset(shapeInfo(), indices);

    for(int i = 0; i < rank; ++i)
        if(minDim > shape[i])
            minDim = shape[i];

    float v = 1.0f;

    for(int i = 0; i < minDim; ++i)
        templatedSet<float>(buffer(), i*offset, this->dataType(), &v);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void templatedSwap(void *xBuffer, void *yBuffer, const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, Nd4jLong length) {
    auto x = reinterpret_cast<T *>(xBuffer);
    auto y = reinterpret_cast<T *>(yBuffer);

    const bool isSameOrders = shape::order(xShapeInfo) == shape::order(xShapeInfo);

    const auto xEws = shape::elementWiseStride(xShapeInfo);
    const auto yEws = shape::elementWiseStride(yShapeInfo);

    auto func = PRAGMA_THREADS_FOR {
        if(isSameOrders && xEws > 0 && yEws > 0) {
            for(auto i = start; i < stop; i++)
                sd::math::nd4j_swap(x[i*xEws], y[i*yEws]);
        }
        else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
            for(auto i = start; i < stop; i++) {
                const auto ind = shape::getIndexOffset(i, xShapeInfo);
                sd::math::nd4j_swap(x[ind], y[ind]);
            }
        }
        else {
            for(auto i = start; i < stop; i++) {
                const auto xInd = shape::getIndexOffset(i, xShapeInfo);
                const auto yInd = shape::getIndexOffset(i, yShapeInfo);
                sd::math::nd4j_swap(x[xInd], y[yInd]);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, length);
}
BUILD_SINGLE_TEMPLATE(template void templatedSwap, (void *xBuffer, void *yBuffer, const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, Nd4jLong length), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::swapUnsafe(NDArray& other) {
    auto xType = this->dataType();

    if (xType != other.dataType())
        throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

    if(buffer() == nullptr || other.buffer() == nullptr)
        throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

    if(lengthOf() != other.lengthOf())
        throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

    BUILD_SINGLE_SELECTOR(xType, templatedSwap, (buffer(), other.buffer(), shapeInfo(), other.shapeInfo(), this->lengthOf()), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
    // no-op
}

void NDArray::prepareSpecialUse(const std::vector<const NDArray*>& writeList, const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
    // no-op
}
void NDArray::registerSpecialUse(const std::vector<const NDArray*>& writeList, const std::vector<const NDArray*>& readList) {
    // no-op
}
void NDArray::preparePrimaryUse(const std::vector<const NDArray*>& writeList, const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
    // no-op
}
void NDArray::registerPrimaryUse(const std::vector<const NDArray*>& writeList, const std::vector<const NDArray*>& readList) {
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
    void* NDArray::specialBufferWithOffset(Nd4jLong offset) {
        return nullptr;
    }

////////////////////////////////////////////////////////////////////////
const void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
    return nullptr;
}

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBuffer() {
    if (_buffer->special() == nullptr)
        return buffer();
    // FIXME: this should be fixed once CUDA backend added
    return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}

////////////////////////////////////////////////////////////////////////
void const* NDArray::specialBuffer() const {
    if (_buffer->special() == nullptr)
        return buffer();
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
            memcpy(&shapeNew[-diff], result.shapeInfo()+1, rankOld * sizeof(Nd4jLong));   // put old shape numbers at rest of positions
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

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                auto yOffset = shape::subArrayOffset(i, newShapeInfo, shapeInfo());
                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign,(result.buffer(), i, this->buffer(), yOffset), LIBND4J_TYPES);
            }
        };

        samediff::Threads::parallel_for(func, 0, resultLen);
    }
    else {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                auto xOffset = result.getOffset(i);
                auto yOffset = shape::subArrayOffset(i, newShapeInfo, shapeInfo());
                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign,(result.buffer(), xOffset, this->buffer(), yOffset), LIBND4J_TYPES);
            }
        };

        samediff::Threads::parallel_for(func, 0, resultLen);
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
    if(!shape::equalsSoft(newShapeInfo, target.shapeInfo()))  {
        delete []newShapeInfo;
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
    }

    // fill newBuff, loop through all elements of newBuff
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const auto targetLen = target.lengthOf();
    if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.buffer(), i, buffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else if(target.ordering() == 'c' && ews > 1) {
        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.buffer(), i*ews, buffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {

        for(Nd4jLong i=0;  i<targetLen; ++i) {

            auto xOffset = target.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.buffer(), xOffset, buffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
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
    if(target.ordering() == 'c' && ews >= 1) {

        for(Nd4jLong i=0;  i<targetLen; ++i) {
            auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.buffer(), i*ews, buffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
    else {

        for(Nd4jLong i=0;  i<targetLen; ++i) {

            auto xOffset = target.getOffset(i);
            auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
            BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign, (target.buffer(), xOffset, buffer(), yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
static void repeat_(const NDArray& input, NDArray& output, const std::vector<int>& repeats, const int axis) {

    const X* x = input.bufferAsT<X>();
          Z* z = output.bufferAsT<Z>();

    const int rank    = input.rankOf();    // xRank = zRank
    const int zLen    = output.lengthOf(); // xLen <= zLen
    const uint repSize = repeats.size();

    // loop through input array
    auto func = PRAGMA_THREADS_FOR {

        int coords[MAX_RANK], temp;

        for (auto i = start; i < stop; i++) {

            shape::index2coordsCPU(start, i, output.shapeInfo(), coords);
            const auto zOffset = shape::getOffset(output.shapeInfo(), coords);

            temp = coords[axis];

            if (repSize > 1) {
                for (uint j = 0; j < repSize; ++j) {
                    coords[axis] -= repeats[j];
                    if (coords[axis] < 0) {
                        coords[axis] = j;
                        break;
                    }
                }
            } else
                coords[axis] /= repeats[0];

            z[zOffset] = x[shape::getOffset(input.shapeInfo(), coords)];

            coords[axis] = temp;
        }
    };

    samediff::Threads::parallel_for(func, 0, zLen);
}

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<int>& repeats) const {

    NDArray output('c', ShapeUtils::evalRepeatShape(axis, repeats, *this), dataType(),  getContext());

    BUILD_SINGLE_SELECTOR_TWICE(dataType(), repeat_, (*this, output, repeats, axis), LIBND4J_TYPES);

    return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
void NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) const {

    if(!target.isSameShape(ShapeUtils::evalRepeatShape(axis, repeats, *this)))
        throw std::invalid_argument("NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of target array!");

    BUILD_DOUBLE_SELECTOR(dataType(), target.dataType(), repeat_, (*this, target, repeats, axis), LIBND4J_TYPES, LIBND4J_TYPES);
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

