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
#include <exceptions/cuda_exception.h>
#include <specials_cuda.h>
#include <loops/special_kernels.h>
#include <PointersManager.h>
#include "../NDArray.hpp"
#include <ConstantShapeHelper.h>

namespace nd4j {

void* NDArray::platformBuffer()             { return specialBuffer();    }
void* NDArray::getPlatformBuffer() const    { return getSpecialBuffer(); }
void NDArray::syncToDevice() const          { _buffer->syncToSpecial();  }
void NDArray::syncToHost() const            { _buffer->syncToPrimary(getContext()); }
void NDArray::tickWriteHost() const         { _buffer->writePrimary();   }
void NDArray::tickWriteDevice() const       { _buffer->writeSpecial();   }
void NDArray::tickReadHost() const          { _buffer->readPrimary();    }
void NDArray::tickReadDevice() const        { _buffer->readSpecial();    }
void NDArray::tickBothActual() const        { _buffer->writePrimary(); _buffer->readSpecial(); }
bool NDArray::isActualOnHostSide() const    { return _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide() const  { return _buffer->isSpecialActual(); }
void NDArray::makeBothBuffersActual() const { if(!isActualOnHostSide()) syncToHost(); if(!isActualOnDeviceSide()) syncToDevice(); }

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
    if (isS())
        throw std::runtime_error("NDArray::setValueInDiagMatrix: you can't use this method on String array!");
    if(rankOf() != 2)
       throw std::runtime_error("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");
    cudaStream_t* stream = getContext()->getCudaStream();
    const auto rows = sizeAt(0);
    const auto cols = sizeAt(1);
    syncToDevice();

    NDArray val = NDArrayFactory::create(value, getContext());
    switch(direction) {
        case 'u':                           // fill upper triangular block
            BUILD_SINGLE_SELECTOR(dataType(), setDiagonalValueUpper, (getSpecialBuffer(), getSpecialShapeInfo(), val, diag, rows, cols,  *stream), LIBND4J_TYPES);
            break;

        case 'l':                           // fill lower triangular block
            BUILD_SINGLE_SELECTOR(dataType(), setDiagonalValueLower, (getSpecialBuffer(), getSpecialShapeInfo(), val, diag, rows, cols, *stream), LIBND4J_TYPES);
            break;
        default:
            throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
    }

    tickWriteDevice();
}
template void NDArray::setValueInDiagMatrix(const double& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const float& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const float16& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const bfloat16& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const Nd4jLong& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const int& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const int16_t& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const uint8_t& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const int8_t& value, const int diag, const char direction);
template void NDArray::setValueInDiagMatrix(const bool& value, const int diag, const char direction);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
    if (isS())
        throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

    if (rankOf() != 2)
        throw std::runtime_error("NDArray::setIdentity: method should work only for 2D tensors. But " + toStringValue(rankOf()) + " was given.");

    this->assign(1.);

    setValueInDiagMatrix(0.f, 1, 'u');
    setValueInDiagMatrix(0.f, -1, 'l');
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NDArray::swapUnsafe(NDArray& other) {
    auto xType = this->dataType();

    if (xType != other.dataType())
        throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

    if(specialBuffer() == nullptr || other.specialBuffer() == nullptr)
        throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

    if(lengthOf() != other.lengthOf())
        throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

    BUILD_SINGLE_SELECTOR(xType, templatedSwapUnsafe, (specialBuffer(), specialShapeInfo(), other.specialBuffer(), other.specialShapeInfo(), getContext()->getCudaStream()), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
    auto res = cudaStreamSynchronize(*(getContext()->getCudaStream()));
    if (res != 0)
        throw std::runtime_error(msg + std::string(": synchronization failed !"));
}
////////////////////////////////////////////////////////////////////////
void NDArray::prepareSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {

    for (const auto& a : readList)
        a->syncToDevice();

    if (synchronizeWritables)
        for (const auto& a : writeList)
            a->syncToDevice();
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {

    for (const auto& p : readList)
        p->tickReadDevice();

    for (const auto& p : writeList)
        p->tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {

    for (const auto& a : readList)
            a->syncToHost();

    if (synchronizeWritables)
        for (const auto& a : writeList)
            a->syncToHost();
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerPrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {

    for (const auto& p : readList)
        p->tickReadHost();

    for (const auto& p : writeList)
        p->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////
void NDArray::syncShape() const {
    cudaMemcpy(getSpecialShapeInfo(), getShapeInfo(), shape::shapeInfoByteLength(getShapeInfo()), cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////////////////
void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
    return getSpecialBuffer() != nullptr ? static_cast<int8_t*>(getSpecialBuffer()) + (offset * sizeOfT()) : nullptr;
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
    std::shared_ptr<DataBuffer> newBuff = std::make_shared<DataBuffer>(shape::length(newShapeInfo) * sizeOfT(), dataType(), getContext()->getWorkspace(), true);
    // assign new shape and new buffer to resulting array
    NDArray result(newBuff, ShapeDescriptor(newShapeInfo), getContext());

    // fill newBuff, loop through all elements of newBuff
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const auto resultLen = result.lengthOf();
    auto xType = this->dataType();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&result}, {this});
    BUILD_SINGLE_SELECTOR(xType, tileKernelH, (this->getSpecialBuffer(), this->getSpecialShapeInfo(), result.getSpecialBuffer(), result.getSpecialShapeInfo(), resultLen, *stream), LIBND4J_TYPES);
    registerSpecialUse({&result}, {this});

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
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const int targetLen = target.lengthOf();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&target}, {this});
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (getSpecialBuffer(), getSpecialShapeInfo(), target.getSpecialBuffer(), target.getSpecialShapeInfo(), targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
    registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) const {
    if(rankOf() > target.rankOf())
        throw std::runtime_error("NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

    if(!ShapeUtils::areShapesBroadcastable(*this, target))
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

    // fill newBuff, loop through all elements of newBuff
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const auto ews = target.ews();
    const auto targetLen = target.lengthOf();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&target}, {this});
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (getSpecialBuffer(), getSpecialShapeInfo(), target.getSpecialBuffer(), target.getSpecialShapeInfo(), targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
    registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
NDArray* NDArray::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {
    auto outShape = ShapeUtils::evalRepeatShape(dimension, repeats, *this);

    // the size of outShape == rank
    int rank = rankOf();            // = outShape.size()

    std::vector<Nd4jLong> newShape(rank);
    for (int i = 0; i < rank; i++)
        newShape[i] = outShape[i];

    auto ret = new NDArray('c', outShape, dataType(),  getContext());

    auto repeatDelta = shape::prodLong(newShape.data(), rank) / this->lengthOf();
    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
    const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(getShapeInfo(), dimsToExclude); //this->tensorsAlongDimension({dimension});
    std::vector<int> copy({dimension});

    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(ret->getShapeInfo(), copy);

    prepareSpecialUse({ret}, {this});
    auto stream = getContext()->getCudaStream();
    BUILD_SINGLE_SELECTOR(dataType(), repeatKernelH, (getSpecialBuffer(), ret->getSpecialBuffer(), numTads, lengthOf(), ret->lengthOf(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), *stream), LIBND4J_TYPES);
    registerSpecialUse({ret}, {this});

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

    std::vector<int> copy({dimension});
    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(target.getShapeInfo(), copy);

    NDArray::prepareSpecialUse({&target}, {this});
    auto stream = getContext()->getCudaStream();
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), repeatKernelHH, (getSpecialBuffer(), target.getSpecialBuffer(), numTads, lengthOf(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), *stream), LIBND4J_TYPES, LIBND4J_TYPES);
    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) const {\

    if(_length == 0)
            { printf("NDArray::printActualBuffer: array length is zero !\n"); return; }

    if(msg)
        printf("%s", msg);

    if(host) {
        if(getBuffer() == nullptr || _length == 0)
            { printf("NDArray::printActualBuffer: host buffer is nullptr !\n"); return; }

        const T* buff = bufferAsT<T>();
        for (uint i = 0; i < _length; i++)
            printf("%.*f, ", precision, (double)buff[getOffset(i)]);
        printf("\n");
    }
    else {
        if(getSpecialBuffer() == nullptr || _length == 0)
            { printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n"); return; }

        void* pHost = operator new(sizeof(T) * _length);

        if (ews() != 1) {
            for (uint i = 0; i < _length; i++)
                cudaMemcpyAsync(pHost + i * sizeof(T), getSpecialBuffer() + getOffset(i) * sizeof(T), sizeof(T), cudaMemcpyDeviceToHost, *(getContext()->getCudaStream()));
        }
        else
            cudaMemcpyAsync(pHost, getSpecialBuffer(), sizeOfT() * _length, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());

        cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
        if(cudaResult != 0)
            throw std::runtime_error("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");

        for (uint i = 0; i < _length; i++)
            printf("%.*f, ", precision, (double)reinterpret_cast<T*>(pHost)[i]);
        printf("\n");

        operator delete(pHost);
    }
}
template void NDArray::printCurrentBuffer<int>(const bool host,const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<float>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<double>(const bool host, const char* msg, const int precision) const;


} // end namespace nd4j
#endif

