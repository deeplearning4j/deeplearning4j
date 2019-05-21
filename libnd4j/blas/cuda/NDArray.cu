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

    prepareSpecialUse({target}, {this, other});

        registerSpecialUse({target}, {this, other});

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
    if (isS())
        throw std::runtime_error("NDArray::setValueInDiagMatrix: you can't use this method on String array!");
    if(rankOf() != 2)
       throw std::runtime_error("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");
    cudaStream_t* stream = _context->getCudaStream();
    const auto rows = sizeAt(0);
    const auto cols = sizeAt(1);
    syncToDevice();

    NDArray val = NDArrayFactory::create(value, _context);
    switch(direction) {
        case 'u':                           // fill upper triangular block
            BUILD_SINGLE_SELECTOR(_dataType, setDiagonalValueUpper, ((void*)_bufferD, _shapeInfoD, val, diag, rows, cols,  *stream), LIBND4J_TYPES);
            break;

        case 'l':                           // fill lower triangular block
            BUILD_SINGLE_SELECTOR(_dataType, setDiagonalValueLower, ((void*)_bufferD, _shapeInfoD, val, diag, rows, cols, *stream), LIBND4J_TYPES);
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

    BUILD_SINGLE_SELECTOR(xType, templatedSwapUnsafe, (specialBuffer(), shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), getContext()->getCudaStream()), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
    auto res = cudaStreamSynchronize(*(_context->getCudaStream()));
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


































    // perform array transformation
    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::FloatOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform FloatOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), DataTypeUtils::pickFloatingType(dataType()), this->_context);

        registerSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformFloat(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        prepareSpecialUse({&result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

        NDArray result(this->_shapeInfo, false, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformSame(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::StrictOps op, void *extraParams) const {
        if (!this->isR())
            throw std::runtime_error("Source array must have one of FLOAT types");

        NDArray result(this->_shapeInfo, false, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformStrict(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), nd4j::DataType::BOOL, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});

        return result;
    }


//////////////////////////////////////////////////////////////////////////
    void NDArray::applyScalarArr(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyScalarArr BoolOps: you can't use this method on String array!");
        if (target == nullptr || !target->isB())
            throw std::invalid_argument("NDArray::applyScalarArr bool method: target is nullptr or has not bool type!");
        if (_dataType != scalar->_dataType) {
            nd4j_printf("This dtype: [%i]; scalar dtype: [%i]\n", this->_dataType, scalar->_dataType);
            throw std::invalid_argument("NDArray::applyScalarArr bool method: this and scalar arrays must have the same type!");
        }

        prepareSpecialUse({target}, {this, scalar});
        NativeOpExecutioner::execScalarBool(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->_buffer, scalar->_shapeInfo, scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()): nullptr);
        registerSpecialUse({target}, {this, scalar});
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::BoolOps op, const T scalar, NDArray *target, ExtraArguments *extraParams) const {

        auto scalarArr = NDArrayFactory::create<T>(scalar, _context);
        applyScalarArr(op, &scalarArr, target, extraParams);
    }

    template <> void NDArray::applyScalar(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) const { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
    template void NDArray::applyScalar<double>(nd4j::scalar::BoolOps op, const double scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<float>(nd4j::scalar::BoolOps op, const float scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<float16>(nd4j::scalar::BoolOps op, const float16 scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<bfloat16>(nd4j::scalar::BoolOps op, const bfloat16 scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<Nd4jLong>(nd4j::scalar::BoolOps op, const Nd4jLong scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int>(nd4j::scalar::BoolOps op, const int scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int16_t>(nd4j::scalar::BoolOps op, const int16_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int8_t>(nd4j::scalar::BoolOps op, const int8_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<uint8_t>(nd4j::scalar::BoolOps op, const uint8_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<bool>(nd4j::scalar::BoolOps op, const bool scalar, NDArray *target, ExtraArguments *extraParams) const;

//////////////////////////////////////////////////////////////////////////
    void NDArray::applyScalarArr(nd4j::scalar::Ops op, const NDArray* scalar, NDArray* target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyScalarArr: you can't use this method on String array!");
        if (!scalar->isScalar())
            throw std::invalid_argument("NDArray::applyScalarArr method: operand is not a scalar!");
        if(target == nullptr)
            target = this;
        if(target->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, scalar->_shapeInfo) && !(target->_dataType == this->_dataType || target->_dataType == scalar->_dataType))
            throw std::invalid_argument("NDArray::applyScalarArr method: wrong type of target array!");

        prepareSpecialUse({target}, {this, scalar});
        NativeOpExecutioner::execScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->getBuffer(), scalar->getShapeInfo(), scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
        registerSpecialUse({target}, {this, scalar});
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::Ops op, const T scalar, NDArray *target, ExtraArguments *extraParams) {

        auto scalarArr = NDArrayFactory::create<T>(this->dataType(), scalar, this->_context);
        applyScalarArr(op, &scalarArr, target, extraParams);
    }

    template <> void NDArray::applyScalar(nd4j::scalar::Ops op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const double scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const float scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const float16 scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const bfloat16 scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const Nd4jLong scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int16_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int8_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const uint8_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const bool scalar, NDArray *target, ExtraArguments *extraParams);

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, const std::vector<int>& dimensions, const NDArray* tadArray, NDArray* target, ExtraArguments* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast: you can't use this method on String array!");
        if(((op == broadcast::Divide || op == broadcast::FloorDiv || op == broadcast::FloorMod) && tadArray->isB()) || (op == broadcast::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyBroadcast: you can't divide by array!");

        if (dimensions.size() == 0)
            return;
        auto result = (NDArray*)this;// == nullptr ? this : target;
        if (target != nullptr)
            result = target;

        if(result->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, tadArray->_shapeInfo))
            throw std::invalid_argument("NDArray::applyBroadcast method: wrong type of target array !");
        if(!result->isSameShape(this))
            throw std::invalid_argument("NDArray::applyBroadcast method: this and target arrays must have the same shape !");

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimensions);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, dimensions);

        auto tadLength = shape::length(packX.primaryShapeInfo());

        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("NDArray::applyBroadcast method: tad length mismatch !");

        NDArray::prepareSpecialUse({result}, {this, tadArray});

        NativeOpExecutioner::execBroadcast(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                           const_cast<int *>(dimensions.data()), (int)dimensions.size(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets());

        registerSpecialUse({result}, {this, tadArray});
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::BoolOps op, const std::vector<int>& dimensions, const NDArray* tadArray, NDArray* target, ExtraArguments* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast BoolOps: you can't use this method on String array!");

        if (dimensions.size() == 0)
            return;

        auto result = target == nullptr ? this : target;

        if(result->_dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: type of target array must be BOOL!");
        if(!result->isSameShape(this))
            throw std::invalid_argument("NDArray::applyBroadcast bool method: this and other arrays must have the same shape !");
        if(_dataType != tadArray->_dataType)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: this and tad arrays must have the same type !");

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimensions);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, dimensions);

        auto tadLength = shape::length(packX.primaryShapeInfo());
        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("Tad length mismatch");

        NDArray::prepareSpecialUse({result}, {this, tadArray});

        NativeOpExecutioner::execBroadcastBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD,
                                               tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD,
                                               result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                               nullptr, (int)dimensions.size(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets());

        registerSpecialUse({result}, {this, tadArray});
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs) const {
        Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, _context->getWorkspace()))          // the rank of new array = max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        NDArray result(newShapeInfo, true, this->_context);

        this->applyTrueBroadcast(op, &other, &result, false, extraArgs);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyIndexReduce(nd4j::indexreduce::Ops op, NDArray* target, const std::vector<int>& dimensions, const ExtraArguments *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        if (target->dataType() != nd4j::DataType::INT64)
            throw std::runtime_error("NDArray::applyIndexReduce operations return INT64");

        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({target}, {this});

        if (target->isScalar()) {
            //target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);
        }
        else {

            std::vector<int> copy(dimensions);
            shape::checkDimensions(rankOf(), copy);

            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());
        }
        registerSpecialUse({target}, {this});
    }

    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    NDArray* NDArray::applyIndexReduce(nd4j::indexreduce::Ops op,const std::vector<int>& dimensions, const ExtraArguments* extraParams ) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::INT64, false, false, _context->getWorkspace());
        auto result = new NDArray(newShape, true, _context);

        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({result}, {this});

        if (rankOf() == copy.size()) {
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD);

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);
        }
        else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,
                                                params,
                                                result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                                nullptr, copy.size(),
                                                packX.specialShapeInfo(), packX.specialOffsets());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);
        }

        registerSpecialUse({result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const ExtraArguments* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyReduce3 method: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");
        // check shapes consistency
        if(!isSameShape(other))
            throw std::runtime_error("NDArray::applyReduce3 method: the shapes of this and other arrays must be the same !");
        // create shapeInfo for scalar
        auto newShape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), _context->getWorkspace());
        // create output array (scalar)
        auto result = new NDArray(newShape, true, _context);
        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({result}, {this, other});

        NativeOpExecutioner::execReduce3Scalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);

        registerSpecialUse({result}, {this, other});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::applyReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");

        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(_dataType), false, false, _context->getWorkspace());
        auto result = new NDArray(newShape, true, _context);
        // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({result}, {this, other});

        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size()) {
            NativeOpExecutioner::execReduce3Scalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);
        }
        else {

            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);
            auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->getShapeInfo(), copy);

            if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()) || (packX.numberOfTads() != packY.numberOfTads() && packX.numberOfTads() != 1 && packY.numberOfTads() != 1))
                throw std::runtime_error("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

            NativeOpExecutioner::execReduce3(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(),  packY.specialShapeInfo(), packY.specialOffsets());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);
        }

        registerSpecialUse({result}, {this, other});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    NDArray* NDArray::applyAllReduce3(nd4j::reduce3::Ops op, const NDArray *other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyAllReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyAllReduce3 method: the types of this and other arrays must be the same !");

        // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(getShapeInfo(), copy);
        auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->getShapeInfo(), copy);

        // check tads shapes
        if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()))
            throw std::runtime_error("NDArray::applyAllReduce3 method: the shapes of array tads are different !");

        // set newShape for output array
        Nd4jLong *newShape = nullptr;
        ALLOCATE(newShape, _context->getWorkspace(), 8, Nd4jLong);
        newShape[0] = 2;        // output rank is always equal to 2 for execAll case
        newShape[1] = packX.numberOfTads();
        newShape[2] = packY.numberOfTads();
        ShapeUtils::updateStridesAndType(newShape, DataTypeUtils::pickFloatingType(_dataType), 'c');
        // create output array
        auto result = new NDArray(newShape, true, _context);
        RELEASE(newShape, _context->getWorkspace());

        NDArray::prepareSpecialUse({result}, {const_cast<NDArray*>(this), const_cast<NDArray*>(other)});

        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NativeOpExecutioner::execReduce3All(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD,result->_buffer,result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                            nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(), packY.specialShapeInfo(), packY.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda failed !", cudaResult);

        registerSpecialUse({result}, {this, other});

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const int* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());

        ShapeDescriptor descriptor(shapeInfo, dataType());
        setShapeInfo(descriptor);

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

        // check if current object is _shapeInfo owner

        auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
        ShapeDescriptor descriptor(shapeInfo, dataType());

        return true;
    }

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension FloatOps cuda: you can't use this method on String array!");
    if (target == nullptr || !target->isR())
        throw std::invalid_argument("NDArray::reduceAlongDimension FloatOps cuda: requires target array to be present and have type form real space!");

    std::vector<int> copy(dimensions);
    if (copy.size())
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension FloatOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension FloatOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension FloatOps cuda failed !", cudaResult);
    }

    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != _dataType)
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: requires target array to be present and have same dtype as input");

    std::vector<int> copy(dimensions);
    if (copy.size())
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension SameOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceSame(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension SameOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::BoolOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: you can't use this method on String array!");
    if (target == nullptr || !target->isB())
        throw std::invalid_argument("NDArray::reduceAlongDimension BoolOps cuda: requires target array to be present and have BOOL type!");

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension BoolOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceBool(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension BoolOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::LongOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != DataType::INT64)
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: requires target array to be present and have type of INT64");

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension LongOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceLong(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension LongOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
    template <typename T>
    void NDArray::p(const Nd4jLong i, const T value) {

        preparePrimaryUse({this}, {}, true);

        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, value): input index is out of array length !");

        auto rp = getOffset(i);
        const void *pV = reinterpret_cast<const void*>(const_cast<T *>(&value));

        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), templatedSet<, T>(this->_buffer, rp, pV), LIBND4J_TYPES);

        registerPrimaryUse({this}, {});
    }

    template void NDArray::p(const Nd4jLong i, const double value);
    template void NDArray::p(const Nd4jLong i, const float value);
    template void NDArray::p(const Nd4jLong i, const float16 value);
    template void NDArray::p(const Nd4jLong i, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const int value);
    template void NDArray::p(const Nd4jLong i, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const bool value);

    ////////////////////////////////////////////////////////////////////////
    void NDArray::p(const Nd4jLong i, const NDArray& scalar) {

        if(!scalar.isScalar())
            throw std::invalid_argument("NDArray::p method: input array must be scalar!");
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");

        preparePrimaryUse({this}, {&scalar}, true);
        auto rp = getOffset(i);
        BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (_buffer, rp, scalar.dataType(), scalar.getBuffer()), LIBND4J_TYPES);
        registerPrimaryUse({this}, {&scalar});
    }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {
        //(*this)(i,j) = value;
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

        preparePrimaryUse({this}, {}, true);

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bool value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");

        preparePrimaryUse({this}, {}, true);

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bool value);

//////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::p(i,j,k,l, value): one of input indexes is out of array length or rank!=4 !");

        preparePrimaryUse({this}, {}, true);

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bool value);

//////////////////////////////////////////////////////////////////////////
    void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
        return _bufferD != nullptr ? _bufferD + (offset * sizeOfT()) : nullptr;
    }

//////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw std::runtime_error("Can't get index higher than total number of TADs");

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        auto array = new NDArray(bufferWithOffset(packX.primaryOffsets()[index]), specialBufferWithOffset(packX.primaryOffsets()[index]), packX.primaryShapeInfo(), _context);
        array->_isView = true;
        array->copyBufferStatus(*this);

        return array;
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::addRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::addRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType) && !(isR() && row->isR() && target->isR()))
            throw std::invalid_argument("NDArray::addRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::subRowVector(const NDArray *row, NDArray * target) const {

        if (isS())
            throw std::runtime_error("NDArray::subRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::subRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::subRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Subtract, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::mulRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::mulRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::mulRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::divRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::divRowVector: you can't use this method on String array!");
        if (row->isB())
            throw std::runtime_error("NDArray::divRowVector: you can't divide by bool row!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::divRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Divide, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    void NDArray::addiRowVector(const NDArray *row) {

        if (isS())
            throw std::runtime_error("NDArray::addiRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::addColumnVector(const NDArray *column, NDArray *target) const {
        if (isS())
            throw std::runtime_error("NDArray::addColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, column->_dataType))
            throw std::invalid_argument("NDArray::addColumnVector: wrong type of target array !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, column});
    }

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    void NDArray::addiColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::addiColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {column});
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    void NDArray::muliColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::muliColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {column});
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
                memcpy(&shapeNew[-diff], result._shapeInfo+1, rankOld * sizeof(Nd4jLong));   // put old shape numbers at rest of positions
                result.reshapei(ordering(), shapeNew);
            }
            return result;             // nothing to do, if diff >= 0 -> identity tile
        }

        // evaluate shapeInfo for resulting array
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
        int8_t * newBuff = nullptr;
        ALLOCATE(newBuff, _context->getWorkspace(), shape::length(newShapeInfo) * sizeOfT(), int8_t);
        // assign new shape and new buffer to resulting array
        NDArray result(newBuff, newShapeInfo, _context, true);

        prepareSpecialUse({&result}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto resultLen = result.lengthOf();
        auto xType = this->dataType();
        auto stream = _context->getCudaStream();
        BUILD_SINGLE_SELECTOR(xType, tileKernelH, (this->_bufferD, this->_shapeInfoD, result._bufferD, result._shapeInfoD, resultLen, *stream), LIBND4J_TYPES);
        registerSpecialUse({&result}, {this});

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::templatedAssign(void *xBuffer, Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        if (xBuffer != nullptr && yBuffer != nullptr)
            *(reinterpret_cast<T*>(xBuffer) + xOffset) = *(reinterpret_cast<T const*>(yBuffer) + yOffset);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES);


    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    void NDArray::tile(const std::vector<Nd4jLong>& reps, NDArray& target) const {

        // evaluate true tile shapeInfo for comparison with target shapeInfo
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
            delete []newShapeInfo;
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
        }

        prepareSpecialUse({&target}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const int ews = target.ews();
        const int targetLen = target.lengthOf();
        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (_bufferD, _shapeInfoD, target._bufferD, target._shapeInfoD, targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
        registerSpecialUse({&target}, {this});
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::tile(NDArray& target) const {
        if(rankOf() > target.rankOf())
            throw std::runtime_error("NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

        if(!ShapeUtils::areShapesBroadcastable(*this, target))
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

        prepareSpecialUse({&target}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto ews = target.ews();
        const auto targetLen = target.lengthOf();
        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (_bufferD, _shapeInfoD, target._bufferD, target._shapeInfoD, targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
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

        auto ret = new NDArray('c', outShape, _dataType,  _context);

        auto repeatDelta = shape::prodLong(newShape.data(), rank) / this->lengthOf();
        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude); //this->tensorsAlongDimension({dimension});
        //tadOnlyInputShapeInfo, tadInputOffsets, tadOnlyOutputShapeInfo, tadOutputOffsets;
        std::vector<int> copy({dimension});

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(ret->getShapeInfo(), copy);

        NDArray::prepareSpecialUse({ret}, {this});

        auto stream = _context->getCudaStream();
        BUILD_SINGLE_SELECTOR(_dataType, repeatKernelH, (_bufferD, ret->_bufferD, numTads, lengthOf(), ret->lengthOf(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), *stream), LIBND4J_TYPES);

        NDArray::registerSpecialUse({ret}, {this});

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
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

        std::vector<int> copy({dimension});
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(target.getShapeInfo(), copy);

        NDArray::prepareSpecialUse({&target}, {this});

        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, repeatKernelHH, (_bufferD, target._bufferD, numTads, lengthOf(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), *stream), LIBND4J_TYPES, LIBND4J_TYPES);

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
            if(_buffer == nullptr || _length == 0)
                { printf("NDArray::printActualBuffer: host buffer is nullptr !\n"); return; }

            const T* buff = bufferAsT<T>();
            for (uint i = 0; i < _length; i++)
                printf("%.*f, ", precision, (double)buff[getOffset(i)]);
            printf("\n");
        }
        else {
            if(_bufferD == nullptr || _length == 0)
                { printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n"); return; }

            void* pHost = operator new(sizeof(T) * _length);

            if (ews() != 1) {
                for (uint i = 0; i < _length; i++)
                    cudaMemcpyAsync(pHost + i * sizeof(T), _bufferD + getOffset(i) * sizeof(T), sizeof(T), cudaMemcpyDeviceToHost, *(_context->getCudaStream()));
            }
            else
                cudaMemcpyAsync(pHost, _bufferD, sizeOfT() * _length, cudaMemcpyDeviceToHost, *_context->getCudaStream());

            cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
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

