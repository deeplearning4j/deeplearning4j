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
#include "../NativeOpExcutioner.h"
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>
#include <MmulHelper.h>
#include <helpers/threshold.h>

namespace nd4j {

    void* NDArray::operator new(size_t i) {
        if (nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            nd4j::memory::Workspace* ws = nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace();

            return ws->allocateBytes((Nd4jLong) i);
        } else {
            auto p = malloc(i);
            
            CHECK_ALLOC(p, "Failed to allocate new NDArray");

            return p;
        }
    }

    void NDArray::operator delete(void* p) {
        if (!nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            free(p);
        }
    }

    NDArray* NDArray::getView() {
        auto view = new NDArray();
        view->_isView = true;
        view->_shapeInfo = _shapeInfo;
        view->_buffer = _buffer;
        view->_workspace = _workspace;
        view->_isShapeAlloc = false;
        view->_isBuffAlloc = false;

        return view;
    }

    template <typename T>
    NDArray* NDArray::asT() {
        auto result  = NDArrayFactory::create_(this->ordering(), this->getShapeAsVector(), DataTypeUtils::fromT<T>());
        auto l = this->lengthOf();

        // FIXME: we want to avoid put/get indexed scalars here really
#pragma omp parallel for
        for (int e = 0; e < l; e++) {
            result->p(e, this->e<T>(e));
        }

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArray::asT, (), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {

    this->_length = shape::length(other._shapeInfo);
    auto shapeLength = shape::shapeInfoByteLength(other._shapeInfo);

    _workspace = other._workspace;
    auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(other._shapeInfo));
    ALLOCATE(_buffer, other._workspace, this->_length * tLen, int8_t);
    ALLOCATE(_shapeInfo, other._workspace, shape::shapeInfoLength(other._shapeInfo), Nd4jLong);

    REPLICATE_SHAPE(other._shapeInfo, this->shapeInfo());

    _isBuffAlloc = true; 
    _isShapeAlloc = true;
    this->assign(&other);
}

////////////////////////////////////////////////////////////////////////
// move constructor
NDArray::NDArray(NDArray&& other) noexcept {

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _workspace    = other._workspace;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;
    this->_length = other.lengthOf();
}

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, nd4j::memory::Workspace* workspace) {
    _buffer    = reinterpret_cast<int8_t *>(buffer);
    _shapeInfo = shapeInfo;
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;
    _workspace = workspace;

    if (shapeInfo != nullptr) {
        _length = shape::length(shapeInfo);
        _dataType = ArrayOptions::dataType(shapeInfo);
    } else
        throw std::runtime_error("NDArray can't be initalized without shapeinfo");
}

////////////////////////////////////////////////////////////////////////
//constructor, create empty array at given workspace
NDArray::NDArray(nd4j::memory::Workspace* workspace) {
    _buffer    = nullptr;
    _shapeInfo = nullptr;
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;
    _workspace = workspace;
    _length = 0;
}

////////////////////////////////////////////////////////////////////////
// default constructor
 NDArray::NDArray() {
        
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));
    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOf(dtype), int8_t);        
    _workspace = workspace;    
    triggerAllocationFlag(true, true);
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));

    if (_length != data.size()) {
        nd4j_printf("NDArray constructor: data size [%i] doesn't match shape length [%i]\n", data.size(), _length);
        throw std::runtime_error("Data size doesn't match shape");
    }

    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOf(dtype), int8_t);        
    _workspace = workspace;    
    triggerAllocationFlag(true, true);

// #pragma omp parallel for simd schedule(static)
//     for(Nd4jLong i=0; i < _length; ++i)
//         templatedSet<double>(_buffer, i, dtype, const_cast<std::vector<double>&>(data).data());

    // memcpy(_buffer, data.data(), _length * sizeof(double));
}


////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {
    
    ALLOCATE(_buffer, workspace, other->_length * DataTypeUtils::sizeOf(other->dataType()), int8_t);
    setShapeInfo(ShapeBuilders::copyShapeInfo(other->_shapeInfo, copyStrides, workspace));
    _workspace = workspace;    
    triggerAllocationFlag(true, true);
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape,  nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {    

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));
        
    _buffer = reinterpret_cast<int8_t *>(buffer);
    _workspace = workspace;
    triggerAllocationFlag(false, true);            
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(const Nd4jLong* shapeInfo, const bool copyStrides, nd4j::memory::Workspace* workspace) {
   
    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");
        
    setShapeInfo(ShapeBuilders::copyShapeInfo(shapeInfo, copyStrides, workspace));

    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOfElement(_dataType) , int8_t);

    memset(_buffer, 0, _length * DataTypeUtils::sizeOfElement(_dataType));

    triggerAllocationFlag(true, true);        
    _workspace = workspace;
}


    template <typename T>
    void NDArray::templatedAssign(void *xBuffer, Nd4jLong xOffset, void *yBuffer, Nd4jLong yOffset) const {
        auto x = reinterpret_cast<T *>(xBuffer);
        auto y = reinterpret_cast<T *>(yBuffer);

        x[xOffset] = y[yOffset];
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedAssign, (void *xBuffer, Nd4jLong xOffset, void *yBuffer, Nd4jLong yOffset) const, LIBND4J_TYPES);

    bool NDArray::isC() const {
        // TODO: this method must be implemented once we add support for complex numbers
        return false;
    }

    bool NDArray::isR() const {
        auto xType = ArrayOptions::dataType(this->_shapeInfo);
        return xType == DataType_FLOAT || xType == DataType_HALF || xType == DataType_DOUBLE || xType == DataType_FLOAT8;
    }

    bool NDArray::isZ() const {
        // TODO: decide if we really want to exclude Bool here
        return !isC() && !isR() && !isB();
    }

    bool NDArray::isB() const {
        return ArrayOptions::dataType(this->_shapeInfo) == DataType_BOOL;
    }


    template<typename T>
    std::string NDArray::toStringValue(T value) {
        std::ostringstream os ;

        //throw the value into the string stream
        os << value ;

        //convert the string stream into a string and return
        return os.str() ;
    }

    template<>
    std::string NDArray::toStringValue(float16 value) {
        std::ostringstream os ;

        //throw the value into the string stream
        os << (float) value ;

        //convert the string stream into a string and return
        return os.str() ;
    }


    std::string NDArray::asIndexedString(Nd4jLong limit) {
        std::ostringstream os;
        os << "[";

        if (limit < 1 || limit > this->lengthOf())
            limit = this->lengthOf();

        for (Nd4jLong e = 0; e < limit; e++) {
            os << toStringValue(this->e<float>(e));

            if (e < limit - 1)
                os << ", ";
        }

        os << "]";

        return os.str();
    }

    std::string NDArray::asString(Nd4jLong limit) {
        std::ostringstream os;
        os << "[";

        if (limit < 1 || limit > this->lengthOf())
            limit = this->lengthOf();

        for (Nd4jLong e = 0; e < limit; e++) {
            os << toStringValue(_buffer[e]);

            if (e < limit - 1)
                os << ", ";
        }

        os << "]";

        return os.str();
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<T> NDArray::getBufferAsVector() {
        std::vector<T> vector(this->lengthOf());

        for (int e = 0; e < this->lengthOf(); e++) {
            vector[e] = this->e<T>(e);
        }

        return vector;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::getBufferAsVector(), LIBND4J_TYPES);



////////////////////////////////////////////////////////////////////////
    std::vector<Nd4jLong> NDArray::getShapeAsVector() {
        std::vector<Nd4jLong> vector(this->rankOf());

        for (int e = 0; e < this->rankOf(); e++)
            vector[e] = this->sizeAt(e);

        return vector;
    }

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeInfoAsFlatVector() {
    int magicNumber = shape::shapeInfoLength(this->rankOf());
    std::vector<int64_t> vector(magicNumber);

    for (int e = 0; e < magicNumber; e++)
        vector[e] = static_cast<int64_t>(_shapeInfo[e]);

    return vector;
}

////////////////////////////////////////////////////////////////////////
    std::vector<Nd4jLong> NDArray::getShapeInfoAsVector() {
        int magicNumber = shape::shapeInfoLength(this->rankOf());
        std::vector<Nd4jLong> vector(magicNumber);

        for (int e = 0; e < magicNumber; e++)
            vector[e] = this->_shapeInfo[e];

        return vector;
    }

////////////////////////////////////////////////////////////////////////
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

        if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) || !this->isSameShape(third)) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }        

        if (this->ordering() == second->ordering() && this->ordering() == third->ordering()  && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == second->ews() && this->ews() == third->ews()) {
#pragma omp parallel for simd schedule(static)
            for (Nd4jLong e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e], second->_buffer[e], third->_buffer[e]);
        } else {
            Nd4jLong tCoord[MAX_RANK];
            Nd4jLong uCoord[MAX_RANK];
            Nd4jLong vCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            #pragma omp parallel for schedule(guided) private(tCoord, uCoord, vCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, this->lengthOf(), tCoord);
                shape::ind2subC(second->rankOf(), second->shapeOf(), e, this->lengthOf(), uCoord);
                shape::ind2subC(third->rankOf(), third->shapeOf(), e, this->lengthOf(), vCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, this->lengthOf(), zCoord);

                auto tOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), tCoord, this->rankOf());
                auto uOffset = shape::getOffset(0, second->shapeOf(), second->stridesOf(), uCoord, second->rankOf());
                auto vOffset = shape::getOffset(0, third->shapeOf(), third->stridesOf(), vCoord, third->rankOf());
                auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[tOffset], second->_buffer[uOffset], third->_buffer[vOffset]);
            }
        }
    }
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<double (double, double, double)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float (float, float, float)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float16 (float16, float16, float16)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int (int, int, int)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int16_t (int16_t, int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint8_t (uint8_t, uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int8_t (int8_t, int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bool (bool, bool, bool)>& func, NDArray* target);


    template<typename T>
    void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<T(T, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw std::runtime_error("Other is null");
        }

        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e], other->_buffer[e]);
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong yCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) private(xCoord, yCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, this->lengthOf(), xCoord);
                shape::ind2subC(other->rankOf(), other->shapeOf(), e, this->lengthOf(), yCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, this->lengthOf(), zCoord);

                auto xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                auto yOffset = shape::getOffset(0, other->shapeOf(), other->stridesOf(), yCoord, other->rankOf());
                auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[xOffset], other->_buffer[yOffset]);
            }
        }
    }
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<double (double, double)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float (float, float)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float16 (float16, float16)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int (int, int)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int16_t (int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint8_t (uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int8_t (int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bool (bool, bool)>& func, NDArray* target);


////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::applyLambda(const std::function<T(T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e]);
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];
#pragma omp parallel for schedule(guided) private(xCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, this->lengthOf(), xCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, this->lengthOf(), zCoord);

                auto xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[xOffset]);
            }
        }
    }
    template void NDArray::applyLambda(const std::function<double(double)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<float(float)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<float16(float16)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<Nd4jLong(Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int16_t(int16_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int8_t(int8_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int(int)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<uint8_t(uint8_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<bool(bool)>& func, NDArray* target);

    template<typename T>
    void NDArray::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func((Nd4jLong) e, this->_buffer[e]);
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) private(xCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, this->lengthOf(), xCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, this->lengthOf(), zCoord);

                auto xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func((Nd4jLong) e, this->_buffer[xOffset]);
            }
        }
    }
    template void NDArray::applyIndexedLambda(const std::function<double(Nd4jLong, double)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<float(Nd4jLong, float)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<float16(Nd4jLong, float16)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<Nd4jLong(Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int(Nd4jLong, int)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int16_t(Nd4jLong, int16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<uint8_t (Nd4jLong, uint8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int8_t(Nd4jLong, int8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<bool(Nd4jLong, bool)>& func, NDArray* target);


    template<typename T>
    void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw std::runtime_error("Other is null");
        }

        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (Nd4jLong e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func((Nd4jLong) e, this->_buffer[e], other->_buffer[e]);
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong yCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) private(xCoord, yCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, this->lengthOf(), xCoord);
                shape::ind2subC(other->rankOf(), other->shapeOf(), e, this->lengthOf(), yCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, this->lengthOf(), zCoord);

                auto xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                auto yOffset = shape::getOffset(0, other->shapeOf(), other->stridesOf(), yCoord, other->rankOf());
                auto zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func((Nd4jLong) e, this->_buffer[xOffset], other->_buffer[yOffset]);
            }
        }
    }
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<double (Nd4jLong, double, double)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float (Nd4jLong, float, float)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float16 (Nd4jLong, float16, float16)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int (Nd4jLong, int, int)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int16_t (Nd4jLong, int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint8_t (Nd4jLong, uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int8_t (Nd4jLong, int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bool (Nd4jLong, bool, bool)>& func, NDArray* target);
#endif



////////////////////////////////////////////////////////////////////////
    std::vector<int8_t> NDArray::asByteVector() {
        std::vector<int8_t> result((unsigned long long) this->lengthOf() * sizeOfT());

        if (this->isView()) {
            auto tmp = this->dup(this->ordering());

            memcpy(result.data(), tmp->_buffer, (unsigned long long) tmp->lengthOf() * sizeOfT());

            delete tmp;
        } else {
            memcpy(result.data(), _buffer, (unsigned long long) this->lengthOf() * sizeOfT());
        }

        return result;
    }

    void NDArray::linspace(const double start) {
        linspace(start, 1);
    }

    void NDArray::linspace(const double start, const double step) {
        Nd4jLong numElements = this->lengthOf();

        for (Nd4jLong e = 0; e < numElements; e++) {
            this->p(e, step * (e + 1));
        }
    }

////////////////////////////////////////////////////////////////////////
    void* NDArray::getBuffer() const {
        return _buffer;
    }

    void* NDArray::buffer() {
        return _buffer;
    }

////////////////////////////////////////////////////////////////////////
    Nd4jLong* NDArray::getShapeInfo() const{
        return _shapeInfo;
    }

    Nd4jLong* NDArray::shapeInfo() {
        return _shapeInfo;
    }

////////////////////////////////////////////////////////////////////////
    void* NDArray::specialBuffer() {
        if (_bufferD == nullptr)
            return _buffer;

        // FIXME: this should be fixed once CUDA backend added
        return _bufferD;
    }

////////////////////////////////////////////////////////////////////////
    Nd4jLong* NDArray::specialShapeInfo() {
        if (_shapeInfoD == nullptr)
            return _shapeInfo;

        // FIXME: this should be fixed once CUDA backend added

        return _shapeInfoD;
    }

////////////////////////////////////////////////////////////////////////
    void NDArray::setSpecialBuffers(void * buffer, Nd4jLong *shape) {
        _bufferD = reinterpret_cast<int8_t *>(buffer);
        _shapeInfoD = shape;
    }

    nd4j::DataType NDArray::dataType() {
        return nd4j::ArrayOptions::dataType(this->shapeInfo());
    }

////////////////////////////////////////////////////////////////////////
// assignment operator
    NDArray& NDArray::operator=(const NDArray& other) {
        nd4j_printf("Move operator...\n","")

	if (this == &other) return *this;

    if (_shapeInfo != nullptr && _buffer != nullptr && shape::equalsSoft(_shapeInfo, other._shapeInfo))
        this->assign(&other);        
        // memcpy(_buffer, other._buffer, arrLength*sizeOfT());               // copy elements of other current array
    else {
        if(_isBuffAlloc && _workspace == nullptr)
            delete []_buffer;
        if(_isShapeAlloc && _workspace == nullptr)
            delete []_shapeInfo;

        this->_length = other.lengthOf();
		auto shapeLength = shape::shapeInfoLength(other.rankOf());

        auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(other._shapeInfo));

        ALLOCATE(_buffer, _workspace, this->_length * tLen, int8_t);
        // memcpy(_buffer, other._buffer, arrLength*sizeOfT());               // copy elements of other current array
        ALLOCATE(_shapeInfo, _workspace, shapeLength, Nd4jLong);
        memcpy(_shapeInfo, other._shapeInfo, shape::shapeInfoByteLength(other.rankOf()));     // copy shape information into new array

        shape::updateStrides(_shapeInfo, other.ordering());

        _isBuffAlloc = true;
        _isShapeAlloc = true;
        this->assign(&other);
    }

    return *this;
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
NDArray& NDArray::operator=(NDArray&& other) noexcept {

    nd4j_printf("Move operator...\n","")

    if (this == &other) 
        return *this;

    if(_isBuffAlloc && _workspace == nullptr)
        delete []_buffer;
    if(_isShapeAlloc && _workspace == nullptr)
        delete []_shapeInfo;

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _workspace    = other._workspace;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;
    _length = other._length;

    return *this;
}

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray& NDArray::operator=(const T scalar) {
        this->assign(scalar);
        return *this;
    }
    template NDArray& NDArray::operator=(const double scalar);
    template NDArray& NDArray::operator=(const float scalar);
    template NDArray& NDArray::operator=(const float16 scalar);
    template NDArray& NDArray::operator=(const Nd4jLong scalar);
    template NDArray& NDArray::operator=(const int scalar);
    template NDArray& NDArray::operator=(const int8_t scalar);
    template NDArray& NDArray::operator=(const uint8_t scalar);
    template NDArray& NDArray::operator=(const int16_t scalar);
    template NDArray& NDArray::operator=(const bool scalar);


void NDArray::replacePointers(void *buffer, Nd4jLong *shapeInfo, const bool releaseExisting ) {
    this->_buffer = reinterpret_cast<int8_t *>(buffer);
    this->_shapeInfo = shapeInfo;

    if (releaseExisting) {
        if (_isShapeAlloc && _workspace == nullptr)
            delete[] _shapeInfo;

        if (_isBuffAlloc && _workspace == nullptr)
            delete[] _buffer;
    }
}


    template <typename X, typename Y>
    void NDArray::templatedDoubleAssign(void *xBuffer, Nd4jLong xOffset, void *yBuffer, Nd4jLong yOffset) const {
        auto x = reinterpret_cast<X *>(xBuffer);
        auto y = reinterpret_cast<Y *>(yBuffer);

        x[xOffset] = static_cast<X>(y[yOffset]);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedDoubleAssign, (void *xBuffer, Nd4jLong xOffset, void *yBuffer, Nd4jLong yOffset) const, LIBND4J_TYPES, LIBND4J_TYPES);

// This method assigns values of given NDArray to this one, wrt order
    void NDArray::assign(NDArray *other) {
        if (this->isScalar() && other->isScalar()) {
            BUILD_DOUBLE_SELECTOR(this->dataType(), other->dataType(), templatedDoubleAssign, (this->buffer(), 0, other->buffer(), 0), LIBND4J_TYPES, LIBND4J_TYPES);
            return;
        } else if (other->isScalar()) {
            NativeOpExcutioner::execScalar(scalar::Copy, this->buffer(), this->shapeInfo(), this->buffer(), this->shapeInfo(), other->buffer(), other->shapeInfo(), nullptr);
            return;
        }

        if (other->lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other->ordering() && this->dataType() == other->dataType() && shape::elementWiseStride(this->_shapeInfo) == 1 && shape::elementWiseStride(other->_shapeInfo) == 1) {
            auto l = lengthOf();
            auto s = sizeOfT();

            memcpy(_buffer, other->_buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner::execPairwiseTransform(1, _buffer, _shapeInfo, other->_buffer, other->_shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }

    void NDArray::assign(const NDArray *other) {
        this->assign(const_cast<NDArray*>(other));
    }

// This method assigns values of given NDArray to this one
    void NDArray::assign(NDArray& other) {

        if (this == &other) 
            return;

        if (this->isScalar() && other.isScalar()) {
            BUILD_DOUBLE_SELECTOR(this->dataType(), other.dataType(), templatedDoubleAssign, (this->buffer(), 0, other.buffer(), 0), LIBND4J_TYPES, LIBND4J_TYPES);
            return;
        } else if (other.isScalar()) {
            NativeOpExcutioner::execScalar(scalar::Copy, this->buffer(), this->shapeInfo(), this->buffer(), this->shapeInfo(), other.buffer(), other.shapeInfo(), nullptr);
            return;
        }
                
        if (other.lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && this->dataType() == other.dataType() && shape::elementWiseStride(_shapeInfo) == 1 && shape::elementWiseStride(other._shapeInfo) == 1) {
            
            memcpy(_buffer, other._buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner::execPairwiseTransform(1, _buffer, _shapeInfo, other._buffer, other._shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }


// This method assigns given value to all elements in this NDArray
    void NDArray::assign(const double value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const float value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const float16 value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const Nd4jLong value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int16_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const uint8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const bool value) {
        // just fire scalar
        auto temp = NDArrayFactory::scalar(value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Copy, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    NDArray* NDArray::detach() {
        if (!isAttached())
            return this;

        void* newBuffer;
        Nd4jLong* newShapeInfo;

        auto l = lengthOf();


        newBuffer = new int8_t[lengthOf() * sizeOfT()];

        if (this->ordering() == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), dataType(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), dataType(), shapeOf());

        auto result = new NDArray(newBuffer, newShapeInfo, nullptr);
        result->_isBuffAlloc = true;
        result->_isShapeAlloc = true;

        auto d1 = this->dataType();
        auto d2 = result->dataType();

        auto s1 = this->sizeOfT();
        auto s2 = result->sizeOfT();

        result->assign(this);

        return result;
    }

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
    NDArray* NDArray::dup(const char newOrder) {
    // op
    Nd4jLong newLength = shape::length(_shapeInfo);
    void* newBuffer;
    Nd4jLong* newShapeInfo;

    char order = newOrder;

    if (order == 'a')
        order = this->ordering();

    if (_workspace == nullptr) {
        newBuffer = new int8_t[newLength * sizeOfT()];

        if (order == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), dataType(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), dataType(), shapeOf());

    } else {
        newBuffer = _workspace->allocateBytes(newLength * sizeOfT());
        newShapeInfo = reinterpret_cast<Nd4jLong *>(_workspace->allocateBytes(shape::shapeInfoByteLength(this->rankOf())));

        if (order == 'f')
            shape::shapeBufferFortran(rankOf(), dataType(), shapeOf(), newShapeInfo);
        else
            shape::shapeBuffer(rankOf(), dataType(), shapeOf(), newShapeInfo);
    }
    // FIXME: we know that EWS is always 1 after dup() result
    newShapeInfo[rankOf() * 2 + 2] = 1;

    auto result = new NDArray(newBuffer, newShapeInfo, _workspace);
    // this values should be set, to avoid memleak
    result->_isBuffAlloc = true;
    result->_isShapeAlloc = true;

    result->assign(this);

    return result;
}

    NDArray NDArray::varianceNumber(nd4j::variance::Ops op, bool biasCorrected) {
        auto res = NDArrayFactory::scalar(this->dataType(), this->_workspace);
        NativeOpExcutioner::execSummaryStats(op,this->getBuffer(), this->getShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), biasCorrected);
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    NDArray NDArray::sumNumber() const {
        auto res = NDArrayFactory::scalar(this->dataType(), this->_workspace);
        NativeOpExcutioner::execReduceScalar(1, _buffer, _shapeInfo, nullptr, res.buffer(), res.shapeInfo());
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    NDArray NDArray::meanNumber() const {
        auto res = NDArrayFactory::scalar(this->dataType(), this->_workspace);
        NativeOpExcutioner::execReduceScalar(0, _buffer, _shapeInfo, nullptr, res.buffer(), res.shapeInfo());
        return res;
    }

//////////////////////////////////////////////////////////////////////////
    bool NDArray::isContiguous() {
        Nd4jLong z = 1;
        int d;
        for(d = this->rankOf() - 1; d >= 0; d--)  {
            if(this->sizeAt(d) != 1) {
                if(this->stridesOf()[d] == z)
                    z *= this->sizeAt(d);
                else
                    return false;
            }
        }
        return true;
    }

    bool NDArray::hasNaNs() {
        return this->reduceNumber(nd4j::reduce::IsNan, nullptr).e<int>(0) > 0;
    }

    bool NDArray::hasInfs() {
        return this->reduceNumber(nd4j::reduce::IsInf, nullptr).e<int>(0) > 0;
    }

    bool NDArray::isFinite() {
        return this->reduceNumber(nd4j::reduce::IsInfOrNan, nullptr).e<int>(0) == 0;
    }

    template <typename T, typename Y>
    void NDArray::templatedSet(void *buffer, const Nd4jLong *indices, void *value) {
        auto t = reinterpret_cast<T *>(buffer);
        auto y = *(reinterpret_cast<Y *>(value));

        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), indices, rankOf());
        t[xOffset] = static_cast<T>(y);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, void *value), LIBND4J_TYPES, LIBND4J_TYPES);

    template <typename T, typename Y>
    void NDArray::templatedSet(void *buffer, const Nd4jLong offset, void *value) {
        auto t = reinterpret_cast<T *>(buffer);
        auto y = *(reinterpret_cast<Y *>(value));

        t[offset] = static_cast<T>(y);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong offset, void *value), LIBND4J_TYPES, LIBND4J_TYPES);


    void NDArray::setWorkspace(memory::Workspace* workspace) {
        this->_workspace = workspace;
    }

    void* NDArray::bufferWithOffset(Nd4jLong offset) {
        BUILD_SINGLE_SELECTOR(this->dataType(), return templatedPointerShift, (this->buffer(), offset), LIBND4J_TYPES);
    }
/*
    template <typename T>
    void NDArray::p(const Nd4jLong* indices, const T value) {
        auto xType = this->dataType();
        BUILD_SINGLE_SELECTOR(xType, templatedSet,(this->_buffer, indices, value), LIBND4J_TYPES);
    }
    */

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
    NDArray* NDArray::reduceAlongDimension(nd4j::reduce::Ops op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

        std::vector<int> copy(dimensions);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size() || copy.empty())
            //result->_buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, nullptr);
            NativeOpExcutioner::execReduceScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, result->buffer(), result->shapeInfo());
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, result->getBuffer(), result->getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
        }

        return result;
    }

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
    NDArray NDArray::reduceAlongDims(nd4j::reduce::Ops op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

        std::vector<int> copy(dimensions);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
        NDArray result(newShape, _workspace);
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size() || copy.empty())
            //result._buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, nullptr);
            NativeOpExcutioner::execReduceScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, result.buffer(), result.shapeInfo());
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, result.getBuffer(), result.getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
            /*
            functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, nullptr, result._buffer,
                                                                        result._shapeInfo, copy.data(), copy.size(),
                                                                        tad.tadOnlyShapeInfo, tad.tadOffsets);
                                                                        */
        }

        return result;
    }

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
    void NDArray::reduceAlongDimension(nd4j::reduce::Ops op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, void *extras) const {

        std::vector<int> copy(dimensions);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo())) {
            nd4j_printf("NDArray::reduceAlongDimension method: wrong target shape!\n", "");
            throw std::runtime_error("NDArray::reduceAlongDimension method: wrong target shape!");
        }
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size() || copy.empty())
            //target->_buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
            NativeOpExcutioner::execReduceScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->buffer(), target->shapeInfo());
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
            /*
            functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, extras, target->_buffer,
                                                                        target->_shapeInfo, copy.data(), copy.size(),
                                                                        tad.tadOnlyShapeInfo, tad.tadOffsets);
            */
        }
    }

// method reduces array by excluding its shapes along axes present in dimensions vector
    NDArray *NDArray::reduceAlongDimension(nd4j::reduce::Ops op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
        return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
	}


//
    NDArray NDArray::reduceNumber(nd4j::reduce::Ops op, void *extraParams) const {
        auto shape = ShapeBuilders::createScalarShapeInfo(this->dataType(), this->_workspace);
        NDArray result(shape, this->_workspace);

        NativeOpExcutioner::execReduceScalar(op, _buffer, _shapeInfo, extraParams, result.buffer(), result.shapeInfo());
        return result;
    }

    Nd4jLong NDArray::indexReduceNumber(nd4j::indexreduce::Ops op, void *extraParams) {
        return NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, extraParams);
    }

// perform array transformation
    void NDArray::applyTransform(nd4j::transform::Ops op, NDArray *target, void *extraParams) {
        NativeOpExcutioner::execTransform(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

// perform array transformation
    void NDArray::applyTransform(nd4j::transform::Ops op, void *extraParams) {
        applyTransform(op, this, extraParams);
    }

    // perform array transformation
    NDArray NDArray::transform(nd4j::transform::Ops op, void *extraParams) const {
        NDArray result(this->_shapeInfo, true, this->_workspace);
        NativeOpExcutioner::execTransform(op, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
    }


// perform pairwise transformation
    void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, NDArray *other, void *extraParams) {
        applyPairwiseTransform(op, other, this, extraParams);
    }

// perform pairwise transformation
    void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, NDArray *other, NDArray *target, void *extraParams) {
        if (other->lengthOf() != target->lengthOf())
            throw std::invalid_argument("NDArray::applyPairwiseTransform method - lengths of arrays are mismatched");

        NativeOpExcutioner::execPairwiseTransform(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams);
    }

/*
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyRandom(nd4j::random::RandomBuffer *buffer, NDArray<T>* y, NDArray<T>* z, T* extraArgs) {
        Nd4jPointer state = (Nd4jPointer) buffer;
        if (y == nullptr && z == nullptr) {
            // we're executing indexed z here
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), extraArgs);
        } else if (y == nullptr && z != nullptr) {
            // XZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        } else if (y != nullptr && z != nullptr) {
            // XYZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        }
    }
    */

    Nd4jLong NDArray::tensorsAlongDimension(std::initializer_list<int> dimensions) const {
        return tensorsAlongDimension(std::vector<int>(dimensions));
    }

    Nd4jLong NDArray::tensorsAlongDimension(const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        return numTads;
    }

    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::initializer_list<int>& dimensions) const {
        return tensorAlongDimension(index, std::vector<int>(dimensions));
    }

    void NDArray::printShapeInfo(const char * msg) const {
        //shape::printShapeInfo(_shapeInfo);
        if (msg == nullptr)
            shape::printShapeInfoLinear(_shapeInfo);
        else {
            int rank = shape::rank(_shapeInfo);
            int lim = shape::shapeInfoLength(rank);
            printf("%s: [", msg);
            for (int i = 0; i < shape::shapeInfoLength(rank); i++) {
                printf("%lld", (long long) _shapeInfo[i]);

                if (i < lim - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        fflush(stdout);
    }

    void NDArray::printBuffer(const char* msg, Nd4jLong limit) {
        if (limit == -1)
            limit = (int) this->lengthOf();

        if (msg != nullptr)
            printf("%s: [", msg);
        else
            printf("[");

        for (Nd4jLong e = 0; e < limit; e++) {
            printf("%f", (float) this->_buffer[e]);
            if (e < limit - 1)
                printf(", ");
        }
        printf("]\n");
        fflush(stdout);
    }

    void NDArray::printIndexedBuffer(const char* msg, Nd4jLong limit) const {
        if (limit == -1)
            limit = (int) this->lengthOf();

        if (msg != nullptr)
            printf("%s [", msg);
        else
            printf("[");
        for (Nd4jLong e = 0; e < limit; e++) {
            printf("%f", this->e<float>(e));
            if (e < limit - 1)
                printf(", ");
        }
        printf("]\n");
        fflush(stdout);
    }

    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw std::runtime_error("Can't get index higher than total number of TADs");

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        // shape::printShapeInfoLinear(tad.tadOnlyShapeInfo);

        void *buffer = nullptr;//this->_buffer + tad.tadOffsets[index];
        auto xType = this->dataType();
        BUILD_SINGLE_SELECTOR(xType, buffer = templatedPointerShift, (this->_buffer, tad.tadOffsets[index]), LIBND4J_TYPES);

        Nd4jLong* shapeInfo;
        if (_workspace == nullptr) {
            shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad.tadOnlyShapeInfo)];
        } else {
            shapeInfo = reinterpret_cast<Nd4jLong *>(_workspace->allocateBytes(shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)));
        }
        std::memcpy(shapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));

        auto array = new NDArray(buffer, shapeInfo, _workspace);
        array->_isBuffAlloc = false;
        array->_isShapeAlloc = true;
        array->_isView = true;

        return array;
    }

    template <typename T>
    void* NDArray::templatedPointerShift(void *buffer, Nd4jLong offset) const {
        return reinterpret_cast<T*>(buffer) + offset;
    }
    BUILD_SINGLE_TEMPLATE(template void* NDArray::templatedPointerShift, (void *buffer, Nd4jLong offset) const, LIBND4J_TYPES);

// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected 
    NDArray* NDArray::transpose() const {
        auto shapeInfoLength = shape::shapeInfoLength(rankOf());
        Nd4jLong* newShapeInfo;

        ALLOCATE(newShapeInfo , _workspace, shapeInfoLength, Nd4jLong);
        memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

        auto newArr = new NDArray(_buffer, newShapeInfo, _workspace);
        newArr->_isShapeAlloc = true;
        newArr->_isBuffAlloc  = false;

        newArr->transposei();

        return newArr;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transp() const {
    int shapeInfoLength = shape::shapeInfoLength(rankOf());
    Nd4jLong* newShapeInfo;

    ALLOCATE(newShapeInfo , _workspace, shapeInfoLength, Nd4jLong);
    memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

    NDArray newArr(_buffer, newShapeInfo, _workspace);
    newArr._isShapeAlloc = true;
    newArr._isBuffAlloc  = false;

    newArr.transposei();

    return newArr;
}


////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected 
    void NDArray::transpose(NDArray& target) const {
        
        auto correctShape = ShapeUtils::evalTranspShapeInfo(*this, _workspace);
        if(!shape::equalsStrict(correctShape, target.getShapeInfo()))
            throw std::runtime_error("NDArray::transpose method: the shapeInfo of target array is wrong !");

    // check whether target has allocated (its own) buffer
    if (target._isBuffAlloc) 
        RELEASE(reinterpret_cast<int8_t *>(target._buffer), target._workspace);

    target._buffer = _buffer;
    // don't forget to indicate that memory for new array was allocated
    target._isBuffAlloc = false;
    target._isView = true;

    RELEASE(correctShape, _workspace);
}


////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed 
    void NDArray::transposei() {
        std::vector<int> perm;
        for (int e = this->rankOf() - 1; e >= 0; e--)
            perm.emplace_back(e);

        this->permutei(perm);
    }

    bool NDArray::equalsTo(NDArray &other, double eps) const {
        return equalsTo(&other, eps);
    }

    void * NDArray::getBufferAsPointer(nd4j::DataType dtype) {
        int8_t *ptr = nullptr;
        ALLOCATE(ptr, _workspace, this->lengthOf() * DataTypeUtils::sizeOfElement(dtype), int8_t);

        // FIXME: if we're going to merge this - move loop into selector
        for (int e = 0; e < this->lengthOf(); e++) {
            BUILD_DOUBLE_SELECTOR(dtype, this->dataType(), templatedDoubleAssign, (ptr, e, this->_buffer, e), LIBND4J_TYPES, LIBND4J_TYPES);
        }
        return ptr;
    }

    // This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
    bool NDArray::equalsTo(const NDArray *other, double eps) const {
        if (this->dataType() != other->dataType())
            return false;

        if (lengthOf() != other->lengthOf()) {
            auto t = lengthOf();
            auto o = other->lengthOf();
            return false;
        }

        // we need to be able to compare [1, len] to [len]
        if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
            // FIXME: do something here?
        } else if (!shape::equalsSoft(_shapeInfo, other->_shapeInfo))
            return false;

        auto extras = NDArrayFactory::scalar(eps);
        auto ptr = extras.getBufferAsPointer(this->dataType());

        auto tmp = NDArrayFactory::scalar<float>(0.0f, this->_workspace);

        // we don't need extraparams for this op
        NativeOpExcutioner::execReduce3Scalar(reduce3::EqualsWithEps, _buffer, _shapeInfo, ptr, other->_buffer, other->_shapeInfo, tmp.buffer(), tmp.shapeInfo());


        RELEASE(reinterpret_cast<int8_t *>(ptr), _workspace);

        if (tmp.e<int>(0) > 0)
            return false;

        return true;
    }


//////////////////////////////////////////////////////////////////////////
    void NDArray::addRowVector(const NDArray *row, NDArray *target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
    void NDArray::subRowVector(const NDArray *row, NDArray * target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::subRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Subtract, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
    void NDArray::mulRowVector(const NDArray *row, NDArray *target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, tad->tadOnlyShapeInfo, tad->tadOffsets);
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::divRowVector(const NDArray *row, NDArray *target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Divide, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);

    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    void NDArray::addiRowVector(const NDArray *row) {
    if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
        throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }


//////////////////////////////////////////////////////////////////////////
    void NDArray::addColumnVector(const NDArray *column, NDArray *target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    void NDArray::addiColumnVector(const NDArray *column) {
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    void NDArray::muliColumnVector(const NDArray *column) {
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }

    template <>
    void NDArray::applyScalar(nd4j::scalar::Ops op, const nd4j::NDArray* scalar, NDArray *target, void *extraParams) const {
        if (target == nullptr)
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar->getBuffer(), scalar->getShapeInfo(), extraParams);
        else
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, scalar->getBuffer(), scalar->getShapeInfo(), extraParams);
    }

    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::Ops op, T scalar, NDArray *target, void *extraParams) const {
        auto temp = NDArrayFactory::scalar<T>(scalar, this->_workspace);

        if (target == nullptr)
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, temp.buffer(), temp.shapeInfo(), extraParams);
        else
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, temp.buffer(), temp.shapeInfo(), extraParams);
    }
    template void NDArray::applyScalar(nd4j::scalar::Ops op, double scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, float scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, float16 scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, Nd4jLong scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, int scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, int16_t scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, int8_t scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, uint8_t scalar, NDArray *target, void *extraParams) const;
    template void NDArray::applyScalar(nd4j::scalar::Ops op, bool scalar, NDArray *target, void *extraParams) const;


    void NDArray::applyScalar(nd4j::scalar::Ops op, NDArray* scalar, NDArray* target, void *extraParams) const {
        if (!scalar->isScalar()) {
            throw std::runtime_error("Operand is not a scalar!");
        }

        if (target == nullptr)
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar->buffer(), scalar->shapeInfo(), extraParams);
        else
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, scalar->buffer(), scalar->shapeInfo(), extraParams);
    }


//////////////////////////////////////////////////////////////////////////
    // calculate strides
    void NDArray::updateStrides(const char order) {
    	shape::updateStrides(_shapeInfo, order);
    }

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
    bool NDArray::reshapei(const char order, const std::initializer_list<Nd4jLong>& shape) {
        std::vector<Nd4jLong> vShape(shape);
        return reshapei(order, vShape);
    }

    bool NDArray::reshapei(const std::initializer_list<Nd4jLong>& shape) {
        return reshapei('c', shape);
    }

    bool NDArray::reshapei(const std::vector<Nd4jLong>& shape) {
        return reshapei('c', shape);
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::enforce(const std::initializer_list<Nd4jLong> &dimensions, char order) {
        std::vector<Nd4jLong> dims(dimensions);
        enforce(dims, order);
    }

    void NDArray::enforce(std::vector<Nd4jLong> &dimensions, char o) {
        Nd4jLong prod = 1;
        for (int e = 0; e < dimensions.size(); e++)
            prod *= dimensions[e];

        if (prod != this->lengthOf()) {
            std::string current = ShapeUtils::shapeAsString(this);
            std::string enforced = ShapeUtils::shapeAsString(dimensions);
            nd4j_printf("Can't enforce new shape, lengths mismatch. Original shape: %s; Requested shape: %s\n", current.c_str(), enforced.c_str());
            throw std::runtime_error("Incompatible shape");
        }

        Nd4jLong *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(dimensions.size()), Nd4jLong);

        char order = o == 'a' ? this->ordering() : o;

        if (order == 'c')
            shape::shapeBuffer(dimensions.size(), dataType(), dimensions.data(), newShape);
        else
            shape::shapeBufferFortran(dimensions.size(), dataType(), dimensions.data(), newShape);

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);

        _shapeInfo = newShape;
        _isShapeAlloc = true;
    }

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
    bool NDArray::reshapei(const char order, const std::vector<Nd4jLong>& cshape) {

    // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary 
    if(order == ordering() && rankOf() == cshape.size()) {
        bool areShapesSame = true;
        for(int i = 0; i < cshape.size(); ++i)
            if(cshape[i] != sizeAt(i)) {
                areShapesSame = false;
                break;
            }
        if(areShapesSame)
            return areShapesSame;        
    }

    std::vector<Nd4jLong> shape(cshape);
    int rank = shape.size();

    // looking for negative in shape

    int numberNegativesOnes = 0;

    Nd4jLong* shape_ = shape.data();
    for (int i = 0; i < (int) shape.size(); i++) {
        if (shape[i] < 0) {
            if (numberNegativesOnes >= 1)
                throw std::runtime_error("Only one dimension can be negative at once");

            numberNegativesOnes++;

            int shapeLength = 1;
            for (int j = 0; j < (int) shape.size(); j++)
                if (i != j)
                    shapeLength *= shape_[j];

            Nd4jLong realShape = nd4j::math::nd4j_abs<int>(lengthOf() / shapeLength);
            auto thisNewShape = new Nd4jLong[shape.size()];

            for (int j = 0; j < (int) shape.size(); j++) 
                if (i != j) 
                    thisNewShape[j] = shape_[j];
                else
                    thisNewShape[j] = realShape;
            
            shape_ = thisNewShape;
        }
    }

    for (int e = 0; e < (int) shape.size(); e++) 
        shape[e] = shape_[e];

    if (numberNegativesOnes > 0)
        delete[] shape_;

    int arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(_buffer==nullptr || arrLength != this->lengthOf()) {
        this->printShapeInfo("Mismatched shape");
        nd4j::Logger::printv("Shape requested: ", shape);
        nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
        throw std::runtime_error("Bad shape!");
    }

    int shapeLength = shape::shapeInfoLength(rank);
    // remember old values

    // we can do this only if there was no permute applied, or there are no weird strides
    if (shape::canReshape(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f')) {
        Nd4jLong *shapeInfoNew;
        ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), Nd4jLong);

        shape::reshapeCF(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f', shapeInfoNew);

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);

        _shapeInfo = shapeInfoNew;
        _isShapeAlloc = true;
    } else {
        Nd4jLong *shapeInfoNew;
        ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), Nd4jLong);

        if (order == 'c')
            shape::shapeBuffer(shape.size(), dataType(), shape.data(), shapeInfoNew);
        else
            shape::shapeBufferFortran(shape.size(), dataType(), shape.data(), shapeInfoNew);

        int8_t *newBuffer;
        ALLOCATE(newBuffer, _workspace, this->lengthOf(), int8_t);

        NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Ops::Copy, newBuffer, shapeInfoNew, this->_buffer, this->_shapeInfo, newBuffer, shapeInfoNew, nullptr);

        if (_isBuffAlloc)
            RELEASE(_buffer, _workspace);


        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);


        _buffer = newBuffer;
        _shapeInfo = shapeInfoNew;
        _isShapeAlloc = true;
        _isBuffAlloc = true;
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::argMax(std::initializer_list<int> dimensions) {
        if (dimensions.size() == 0) {
            Nd4jLong max = 0;
            auto mv = -DataTypeUtils::max<float>();
            for (Nd4jLong e = 0; e < this->lengthOf(); e++) {
                auto val = this->e<float>(e);
                if (mv < val) {
                    mv = val;
                    max = e;
                }
            }

            return max;
        } else
            throw std::runtime_error("Not implemented yet");
    }

    //////////////////////////////////////////////////////////////////////////
    // create new array with corresponding order and shape, new array will point to the same _buffer as this array
    NDArray* NDArray::reshape(const char order, const std::vector<Nd4jLong>& shape) const {
	    int shapeInfoLength = shape::shapeInfoLength(rankOf());
	    Nd4jLong* newShapeInfo = nullptr;

	    ALLOCATE(newShapeInfo , _workspace, shapeInfoLength, Nd4jLong);
	    memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

	    auto newArr = new NDArray(_buffer, newShapeInfo, _workspace);
	    newArr->_isShapeAlloc = true;
	    newArr->_isBuffAlloc  = false;
	    newArr->reshapei(order, shape);

	    return newArr;
    }

    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    void NDArray::tilei(const std::vector<Nd4jLong>& reps) {

        *this = this->tile(reps);
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
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _workspace);
        // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
        int8_t * newBuff = nullptr;
        ALLOCATE(newBuff, _workspace, shape::length(newShapeInfo) * sizeOfT(), int8_t);
        // assign new shape and new buffer to resulting array
        NDArray result(newBuff, newShapeInfo, _workspace);
        result._isShapeAlloc = true;
        result._isBuffAlloc = true;

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto resultLen = result.lengthOf();
        auto xType = this->dataType();
        if(result.ordering() == 'c') {           //  ews == 1 always here
//#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(Nd4jLong i=0;  i<resultLen; ++i) {
                auto yOffset = shape::subArrayIndex(newShapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (newBuff, i, this->_buffer, yOffset), LIBND4J_TYPES);

            }
        }
        else {
            Nd4jLong idx[MAX_RANK];
            auto resultShape   = result.shapeOf();
            auto resultStrides = result.stridesOf();
            const auto resultRank = result.rankOf();
#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
            for(int i=0;  i<resultLen; ++i) {
                shape::ind2subC(resultRank, resultShape, i, resultLen, idx);
                auto xOffset = shape::getOffset(0, resultShape, resultStrides, idx, resultRank);
                auto yOffset = shape::subArrayIndex(newShapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (newBuff, xOffset, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
              
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    void NDArray::tile(const std::vector<Nd4jLong>& reps, NDArray& target) const {
        
        // evaluate true tile shapeInfo for comparison with target shapeInfo
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _workspace);
        if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
            delete []newShapeInfo;
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
        }
        RELEASE(newShapeInfo, _workspace);

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const int ews = target.ews();
        const int targetLen = target.lengthOf();
        auto targetBuff = target.getBuffer();
        auto xType = this->dataType();
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(Nd4jLong i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, i, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, i * ews, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
        else {
            Nd4jLong idx[MAX_RANK];
            auto targetShape     = target.shapeOf();
            auto targetStrides   = target.stridesOf();
            const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
            for(int i=0;  i<targetLen; ++i) {
                shape::ind2subC(targetRank, targetShape, i, targetLen, idx);
                auto xOffset = shape::getOffset(0, targetShape, targetStrides, idx, targetRank);
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, xOffset, this->_buffer, yOffset), LIBND4J_TYPES);
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
        auto targetBuff = target.getBuffer();
        auto xType = this->dataType();
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for (int i = 0; i < targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, i, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, i * ews, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
        else {
            Nd4jLong idx[MAX_RANK];
            auto targetShape     = target.shapeOf();
            auto targetStrides   = target.stridesOf();
            const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
            for(int i=0;  i<targetLen; ++i) {
                shape::ind2subC(targetRank, targetShape, i, targetLen, idx);
                auto xOffset = shape::getOffset(0, targetShape, targetStrides, idx, targetRank);
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_SINGLE_SELECTOR(xType, templatedAssign, (targetBuff, xOffset, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::sizeAt(const int dim) const {
        if (dim >= this->rankOf() || dim < -this->rankOf())
            throw std::runtime_error("Bad size index requested");

        if (dim >= 0)
            return this->_shapeInfo[1+dim];
        else
            return this->_shapeInfo[1+(this->rankOf() + dim)];
    }


    //////////////////////////////////////////////////////////////////////////
    // create new  array by repeating it the number of times given by reps
    NDArray* NDArray::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {
        auto outShape = ShapeUtils::evalRepeatShape(dimension, repeats, *this);
    
        // the size of outShape == rank
        int rank = rankOf();            // = outShape.size()

        auto newShape = new Nd4jLong[rank];
        for (int i = 0; i < rank; i++)
            newShape[i] = outShape[i];

        auto ret = new NDArray('c', outShape, this->dataType(),  _workspace);

        auto repeatDelta = shape::prodLong(newShape, rank) / this->lengthOf();
        auto numTads = this->tensorsAlongDimension({dimension});
        for (int i = 0; i < numTads; i++) {
            auto thisTensor = this->tensorAlongDimension(i, {dimension});
            auto retTensor = ret->tensorAlongDimension(i, {dimension});
            int retIdx = 0;
            if (isR()) {
                for (int k = 0; k < thisTensor->lengthOf(); k++) {
                    auto s = thisTensor->e<double>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor->p<double>(retIdx++, s);
                    }
                }
            } else {
                for (int k = 0; k < thisTensor->lengthOf(); k++) {
                    auto s = thisTensor->e<Nd4jLong>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor->p<Nd4jLong>(retIdx++, s);
                    }
                }
            }

            delete thisTensor;
            delete retTensor;
        }

        delete[] newShape;

        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    // fill array by repeating it the number of times given by reps
    void NDArray::repeat(int dimension, NDArray& target) const {
    
        Nd4jLong repeatDelta = shape::prodLong(target.shapeOf(), rankOf()) / this->lengthOf();
        Nd4jLong numTads = this->tensorsAlongDimension({dimension});
        for (int i = 0; i < numTads; i++) {
            auto thisTensor = this->tensorAlongDimension(i, {dimension});
            auto retTensor = target.tensorAlongDimension(i, {dimension});
            int retIdx = 0;
            if (isR()) {
                for (int k = 0; k < thisTensor->lengthOf(); k++) {
                    auto s = thisTensor->e<double>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor->p<double>(retIdx++, s);
                    }
                }
            } else {
                for (int k = 0; k < thisTensor->lengthOf(); k++) {
                    auto s = thisTensor->e<Nd4jLong>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor->p<Nd4jLong>(retIdx++, s);
                    }
                }
            }

            delete thisTensor;
            delete retTensor;
        }
    }


    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const int* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _workspace);
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, dimensions);
        }

        return true;
    }

    bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _workspace);
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, dimensions);
        }

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const std::initializer_list<int>& dimensions) {
        std::vector<int> vec(dimensions);
        return permutei(vec);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const std::vector<int>& dimensions) {
        return permutei(dimensions.data(), dimensions.size());
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const std::initializer_list<Nd4jLong>& dimensions) {
        std::vector<Nd4jLong> vec(dimensions);
        std::vector<int> ivec(dimensions.size());

        for (int e = 0; e < vec.size(); e++)
            ivec[e] = static_cast<int>(vec[e]);

        return permutei(ivec);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const std::vector<Nd4jLong>& dimensions) {
        std::vector<int> ivec(dimensions.size());

        for (int e = 0; e < dimensions.size(); e++)
            ivec[e] = dimensions[e];

        return permutei(ivec.data(), ivec.size());
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const int* dimensions, const int rank) const {
        // evaluate shapeInfo for output (permuted) array ret
        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _workspace);
        // create array to be returned
        auto ret = new NDArray(_buffer, shapeInfoNew, _workspace);
        // don't forget to indicate that memory for new array was allocated
        ret->_isBuffAlloc = false;
        ret->_isShapeAlloc = true;
	    ret->_isView = true;

        return ret;
    }

    /////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const Nd4jLong* dimensions, const int rank) const {
        int tempDims[MAX_RANK];
        shape::convertT<Nd4jLong, int>(const_cast<Nd4jLong *>(dimensions), tempDims, rank);
        return permute(tempDims, rank);
    }


    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const std::vector<int>& dimensions) const {
        return permute(dimensions.data(), dimensions.size());
    }

    NDArray* NDArray::permute(const std::vector<Nd4jLong>& dimensions) const {
        return permute(dimensions.data(), dimensions.size());
    }


    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const std::initializer_list<int>& dimensions) const {
        std::vector<int> vec(dimensions);
        return permute(vec);
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const std::initializer_list<Nd4jLong>& dimensions) const {
        std::vector<Nd4jLong> vec(dimensions);
        return permute(vec);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::permute(const int* dimensions, const int rank, NDArray& target) const {
        if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
            throw std::runtime_error("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

        // check whether target has allocated (its own) buffer
        if (target._isBuffAlloc)
            RELEASE(target._buffer, target._workspace);

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._workspace);

        target._buffer = _buffer;
        target._shapeInfo = shapeInfoNew;
        // don't forget to indicate that memory for new array was allocated
        target._isBuffAlloc = false;
        target._isShapeAlloc = true;
        //target._isView = true;
    }

    void NDArray::permute(const Nd4jLong *dimensions, const int rank, NDArray& target) const {
        if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
            throw std::runtime_error("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

        // check whether target has allocated (its own) buffer
        if (target._isBuffAlloc)
            RELEASE(target._buffer, target._workspace);

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._workspace);

        target._buffer = _buffer;
        target._shapeInfo = shapeInfoNew;
        // don't forget to indicate that memory for new array was allocated
        target._isBuffAlloc = false;
        target._isShapeAlloc = true;
        //target._isView = true;

    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::permute(const std::vector<int>& dimensions, NDArray& target) const {
        permute(dimensions.data(), dimensions.size(), target);
    }

    void NDArray::permute(const std::vector<Nd4jLong>& dimensions, NDArray& target) const {
        permute(dimensions.data(), dimensions.size(), target);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, std::initializer_list<int> dimensions, const NDArray* tadArray, NDArray* target, void* extraArgs) {
        std::vector<int> vec(dimensions);
        applyBroadcast(op, vec, tadArray, target, extraArgs);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, std::vector<int>& dimensions, const NDArray* tadArray, NDArray* target, void* extraArgs) {
        if (dimensions.size() == 0)
            return;

        std::vector<int> copy(dimensions);

        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("Tad length mismatch");

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        auto result = target == nullptr ? this : target;

        // TODO: eventually we want separate tads here
        NativeOpExcutioner::execBroadcast(op, this->_buffer, this->_shapeInfo, tadArray->_buffer, tadArray->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, tad.tadOnlyShapeInfo, tad.tadOffsets);
    }


    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast method: target or other = nullptr !");

        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op.p, const_cast<NDArray*>(other), extraArgs);
            return;
        }
        if (other->isScalar()) {
            this->applyScalar(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = this;
            min = other;
        }
        else {
            max = other;
            min = this;
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsSoft(target->getShapeInfo(), newShapeInfo))
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape of target array is wrong !");

            // if workspace is not null - do not call delete.
            if (_workspace == nullptr)
                delete[] newShapeInfo;
        }

        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *target);
        }
        else
            target->assign(max);


        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            target->applyBroadcast(op.b, sameDims, pMin, nullptr, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);
        
#pragma omp parallel for schedule(guided)
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto targetSubArr = (*target)(i, dimsToExclude);
                pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
            }
        }

        if(pMin != min)
            delete pMin;
    }

//////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, void *extraArgs) const {
        Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, *other, true, newShapeInfo, _workspace))          // the rank of new array = max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        auto result = new NDArray(newShapeInfo, false, this->_workspace);

        // if workspace is not null - do not call delete.
        if (_workspace == nullptr)
            delete[] newShapeInfo;

        this->applyTrueBroadcast(op, other, result, false, extraArgs);
  
        return result;
    }


    //////////////////////////////////////////////////////////////////////////
    NDArray NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, void *extraArgs) const {
        auto pResult = this->applyTrueBroadcast(op, &other, extraArgs);
        pResult->_isShapeAlloc = false;
        pResult->_isBuffAlloc  = false;

        NDArray result(pResult->_buffer, pResult->_shapeInfo, _workspace);
        result._isShapeAlloc = true;
        result._isBuffAlloc  = true;
    
        delete pResult;

        return result;
    }


    //////////////////////////////////////////////////////////////////////////
    // return array which is broadcasted from this and argument array
    NDArray* NDArray::broadcast(const NDArray& other) {
	    // the orders must be the same
	    char order = ordering();
	    if(order != other.ordering())
		    throw std::runtime_error("Broadcast method: arrays have different orders!");

	    // recognize shapes with smaller and bigger rank
	    Nd4jLong* biggerShapeInfo = nullptr;
	    Nd4jLong* smallerShapeInfo = nullptr;
	    int smallerRank, biggerRank;
	    if (rankOf() > other.rankOf()) {
		    biggerShapeInfo = _shapeInfo;
		    biggerRank = shape::rank(_shapeInfo);
		    smallerShapeInfo = other._shapeInfo;
		    smallerRank = shape::rank(other._shapeInfo);
	    }
	    else {
		    biggerShapeInfo = other._shapeInfo;
		    biggerRank = shape::rank(other._shapeInfo);
		    smallerShapeInfo = _shapeInfo;
		    smallerRank = shape::rank(_shapeInfo);
	    }

	    // check shapes on consistency
	    int diff = biggerRank - smallerRank;
	    for (int i = smallerRank; i<=1; --i)
		    if(biggerShapeInfo[diff+i] != smallerShapeInfo[i] && biggerShapeInfo[i] != 1 && smallerShapeInfo[i] != 1)
			    throw std::runtime_error("Broadcast method: arrays have incompatible shapes !");

		// create and fill ret shapeInfo
	    auto shapeInfoNew = new Nd4jLong[shape::shapeInfoLength(biggerRank)];
	    memcpy(shapeInfoNew, biggerShapeInfo, shape::shapeInfoByteLength(biggerRank));
	    for (int i = smallerRank; i>=1; --i)
		    if(shapeInfoNew[diff+i] == 1 || smallerShapeInfo[i] == 1)
			    shapeInfoNew[diff+i] *= smallerShapeInfo[i];

	    auto ret = new NDArray(shapeInfoNew, _workspace);
	    ret->updateStrides(order);
	    delete []shapeInfoNew;

    	return ret;
    }


    //////////////////////////////////////////////////////////////////////////
    // check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
    bool NDArray::hasOrthonormalBasis(const int arg) {
	    if(rankOf() !=2 )
		    throw std::runtime_error("NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !");

	    if(arg!=0  && arg!=1)
		    throw std::runtime_error("NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !");

	    const double eps = 1e-5;
        double dot = 0.f;

        if(arg) {					// check whether columns create orthogonal basis
		    for(int j=0; j<columns()-1; ++j)
			    for(int k=j+1; k<columns(); ++k) {
				    for(int i=0; i<rows(); ++i)
					    dot += e<double>(i,j)*e<double>(i,k);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.f;
			    }

			    for(int j=0; j<columns(); ++j)	{	// check whether norm of column vector = 1
			        for(int i=0; i<rows(); ++i)
				        dot += e<double>(i,j)*e<double>(i,j);
			    if(dot != 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.f) > eps)
				    return false;

			    dot = 0.f;
		    }
	    }
	    else {						// check whether rows create orthogonal basis
		    for(int i=0; i<rows()-1; ++i)
			    for(int k=i+1; k<rows(); ++k) {
				    for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(k,j);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.;
			    }

		        for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			        for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(i,j);

			        if(dot!= 0. && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.) > eps)
				        return false;
			        dot = 0.;
		        }
	        }
	    return true;
    }

    template <typename T>
    std::vector<T> NDArray::asVectorT() {
        std::vector<T> result(this->lengthOf());

#pragma omp parallel for simd
        for (int e = 0; e < this->lengthOf(); e++)
            result[e] = this->e<T>(e);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::asVectorT(), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    // check whether array is identity matrix
    bool NDArray::isIdentityMatrix() {
	    if(rankOf() !=2 || rows() != columns())
		    throw std::runtime_error("isIdentityMatrix method: matrix must be square and have rank = 2 !");

	    const double eps = 1e-5f;
	    for(int i=0; i<rows(); ++i)
		    if(nd4j::math::nd4j_abs(e<double>(i,i) - 1.f) > eps)
			    return false;

	    for(int i=0; i<rows(); ++i)
		    for(int j=0; j<columns(); ++j) {
			    if (i == j)
			        continue;

			    if(nd4j::math::nd4j_abs(e<double>(i,j)) > eps)
				    return false;
		    }
    	return true;
    }

    //////////////////////////////////////////////////////////////////////////
    // check whether array is unitary matrix
    bool NDArray::isUnitary() {
        if(rankOf() !=2 || rows() != columns())
            throw std::runtime_error("isUnitary method: matrix must be square and have rank = 2 !");

        auto tr = this->transpose();
        auto trMul = MmulHelper::mmul(this, tr, nullptr, 1.f, 0.f);

        bool result = trMul->isIdentityMatrix();
        delete tr;
        delete trMul;
    
        return result;
    }


//////////////////////////////////////////////////////////////////////////
    template <typename T>
    T NDArray::e(const Nd4jLong i) const {
        auto xType = this->dataType();
        // return (*this)(i);
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, i), LIBND4J_TYPES);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j) const {
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");

        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        //return (*this)(i, j);
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {
        //return (*this)(i, j, k);
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || j >= shapeOf()[2])
            throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");

        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

    // returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l) const {
        //return (*this)(i, j, k);
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || j >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=4 !");

        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
    template <typename T>
    void NDArray::p(const Nd4jLong i, const T value) {
        auto xType = this->dataType();

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, i, p), LIBND4J_TYPES);
    }
    template void NDArray::p(const Nd4jLong i, const double value);
    template void NDArray::p(const Nd4jLong i, const float value);
    template void NDArray::p(const Nd4jLong i, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const int value);
    template void NDArray::p(const Nd4jLong i, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const bool value);


    template <typename T>
    T* NDArray::bufferAsT() {
        return reinterpret_cast<T*>(_buffer);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template, * NDArray::bufferAsT(), LIBND4J_TYPES);


    void NDArray::p(const Nd4jLong i, const NDArray& value) {
        // probqbly wrong args order
        BUILD_SINGLE_SELECTOR(value.dataType(), templatedSet, (_buffer, i, value.dataType(), value.getBuffer()), LIBND4J_TYPES);
        // void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, void *value)
    }


//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {
        //(*this)(i,j) = value;

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float16 value);
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
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bool value);

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value) {
        //(*this)(i,j,k) = value;
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bool value);


    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(IndicesList& idx, std::vector<Nd4jLong>& strides) const {
        auto raw = subarray(idx);

        for (int e = 0; e < strides.size(); e++)
            raw->stridesOf()[e] *= strides[e];

        return raw;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(IndicesList& idx) const {
        auto xType = this->dataType();
        // scalar subarray edge case
        if (idx.isScalar()) {
            auto pnt = idx.at(0)->getIndices().at(0);
            //return new NDArray('c', {1, 1}, {this->e(pnt)});
            BUILD_SINGLE_SELECTOR(xType, return NDArrayFactory::scalar_, (this->e<float>(pnt), this->_workspace), LIBND4J_TYPES);
        }

        if (idx.size() != this->rankOf())
            throw std::runtime_error("Number of indices should match");

        Nd4jLong* newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        auto shapeOf = shape::shapeOf(newShape);
        auto stridesOf = shape::stride(newShape);

        Nd4jLong offset = 0;

        //shape::printShapeInfoLinear(newShape);

        for (int d = 0; d < idx.size(); d++) {
            // building new shape first
            auto index = idx.at(d);
            if (index->isAll()) {
                // shape is unchanged  for this dimension
            } else {
                // size at this dimension equals to either 1 or interval
                shapeOf[d] = index->getIndices().size();

                // for offset we're taking only the first index
                auto first = index->getIndices().at(0);
                offset += first * stridesOf[d];

                shape::stride(newShape)[d] *= index->stride();
                nd4j_debug("dimension_ [%i] stride [%i]\n", d, index->stride());
            }
        }

        //shape::printShapeInfoLinear(newShape);

        auto result = new NDArray(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(const std::initializer_list<NDIndex*>& idx) const {
        if (idx.size() != this->rankOf())
            throw std::runtime_error("NDArray::subarray: number of indices should match the array rank");

        Nd4jLong *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        auto shapeOf = shape::shapeOf(newShape);
        auto stridesOf = shape::stride(newShape);

        Nd4jLong offset = 0;
        int d = 0;
        for (const auto& index : idx) {
            // building new shape first
            if (index->isAll()) {
                // shape is unchanged  for this dimension
            } else {
                // size at this dimension equals to either 1 or interval
                shapeOf[d] = index->getIndices().size();
                // for offset we're taking only the first index
                auto first = index->getIndices().at(0);
                offset += first * stridesOf[d];
            }
            ++d;
        }

        auto result = new NDArray(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        for (auto v: idx) {
            delete v;
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(const Intervals& idx) const {
        if (idx.size() != this->rankOf())
            throw std::runtime_error("NDArray::subarray: number of indices should match the rank of array!");

        Nd4jLong *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        auto shapeOf = shape::shapeOf(newShape);
        auto stridesOf = shape::stride(newShape);

        Nd4jLong offset = 0;
        for (int d = 0; d < idx.size(); ++d) {
            // building new shape first
            if (!idx[d].empty()) {
                if (idx[d].size() != 2)
                    throw std::runtime_error("NDArray::subarray: the interval must contain only two numbers {first, last} !");
                shapeOf[d] = idx[d][1] - idx[d][0];
                // for offset we're taking only the first index
                offset += idx[d][0] * stridesOf[d];
            }
        }

        auto result = new NDArray(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }


    NDArray* NDArray::asT(DataType dtype) {
        BUILD_SINGLE_SELECTOR(dtype, return asT, (), LIBND4J_TYPES);
    }

    void NDArray::cast(NDArray* target, DataType dtype) {
        // TODO: to be implemented properly
        target->assign(this);
    }

    void NDArray::applyIndexReduce(nd4j::indexreduce::Ops op, const NDArray* target, const std::vector<int>& dimensions, const void *extraParams) const {
        if (target->dataType() != nd4j::DataType::DataType_INT64)
            throw std::runtime_error("IndexReduce operations return INT64");

        if (target->isScalar()) {
            //target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
            reinterpret_cast<Nd4jLong *>(target->_buffer)[0] = NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, const_cast<void *>(extraParams));
        } else {
            std::vector<int> copy(dimensions);
            if (dimensions.size() > 1)
                std::sort(copy.begin(), copy.end());

            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            NativeOpExcutioner::execIndexReduce(op, _buffer, _shapeInfo, const_cast<void *>(extraParams),
                                                                          reinterpret_cast<Nd4jLong *>(target->_buffer),
                                                                          target->_shapeInfo, copy.data(), copy.size(),
                                                                          tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
    }
    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    NDArray* NDArray::applyIndexReduce(nd4j::indexreduce::Ops op,const std::vector<int>& dimensions, const void* extraParams ) const {

        std::vector<int> copy(dimensions);
        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);

        if (rankOf() == copy.size()) {
            const auto res = NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, const_cast<void *>(extraParams));
            result->assign(res);
        } else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            NativeOpExcutioner::execIndexReduce(op, _buffer, _shapeInfo, const_cast<void *>(extraParams),
                                                                    reinterpret_cast<Nd4jLong *>(result->_buffer),
                                                                    result->_shapeInfo, copy.data(), copy.size(),
                                                                    tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const void* extraParams) const {
        // check shapes consistency
        if(!isSameShape(other))
            throw std::runtime_error("NDArray::applyReduce3 method: the shapes of array must be the same !");
        // create shapeInfo for scalar
        Nd4jLong* newShape;
        ALLOCATE(newShape, _workspace, 8, Nd4jLong);
        newShape[0] = 2;    // set rank    
        newShape[1] = 1;    // set first dimension (scalar)
        newShape[2] = 1;    // set second dimension (scalar)
        shape::updateStrides(newShape, 'c');
        // create output array (scalar)
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
     /*
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        
        result->_buffer[0] = functions::reduce3::Reduce3<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), other->_buffer, other->_shapeInfo);        

        delete []extraParamsVals;
     */
        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    NDArray*  NDArray::applyAllReduce3(nd4j::reduce3::Ops op, const NDArray *other, const std::vector<int>& dimensions, const void* extraParams) const {
        // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               
        // create tads
        shape::TAD tadX(_shapeInfo, copy.data(), copy.size());
        tadX.createTadOnlyShapeInfo();
        tadX.createOffsets();

        shape::TAD tadY(other->_shapeInfo, copy.data(), copy.size());
        tadY.createTadOnlyShapeInfo();
        tadY.createOffsets();        
        // check tads shapes
        if(!shape::equalsSoft(tadX.tadOnlyShapeInfo, tadY.tadOnlyShapeInfo)) 
            throw std::runtime_error("NDArray::applyAllReduce3 method: the shapes of array tads are different !");
        // evaluate numbers of tads
        Nd4jLong tadLengthX = shape::tadLength(_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTadsX = lengthOf() / tadLengthX;
        
        Nd4jLong tadLengthY = shape::tadLength(other->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTadsY = other->lengthOf() / tadLengthY;
        // set newShape for output array        
        Nd4jLong *newShape = nullptr;
        ALLOCATE(newShape, _workspace, 8, Nd4jLong);
        newShape[0] = 2;        // output rank is always equal to 2 for execAll case
        newShape[1] = numTadsX;
        newShape[2] = numTadsY;
        shape::updateStrides(newShape, 'c');
        // create output array
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        double* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new double[3] { 0.0, 0.0, 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        NativeOpExcutioner::execReduce3All(op, _buffer, _shapeInfo, const_cast<void *>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size(), tadX.tadOnlyShapeInfo, tadX.tadOffsets, tadY.tadOnlyShapeInfo, tadY.tadOffsets);
        delete []extraParamsVals;
        return result;
    }
 
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const void* extraParams) const {
        
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        double* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new double[3] {0.0, 0.0, 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size())
            NativeOpExcutioner::execReduce3Scalar(op, _buffer, _shapeInfo, const_cast<void *>(extraParams), other->_buffer, other->_shapeInfo, result->_buffer, result->shapeInfo());
        else {
            shape::TAD tadX(_shapeInfo, copy.data(), copy.size());
            tadX.createTadOnlyShapeInfo();
            tadX.createOffsets();

            shape::TAD tadY(other->_shapeInfo, copy.data(), copy.size());
            tadY.createTadOnlyShapeInfo();
            tadY.createOffsets();        
        
            NativeOpExcutioner::execReduce3(op, _buffer, _shapeInfo, const_cast<void *>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size());
        }
        
        delete []extraParamsVals;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());
            
        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        auto result = new NDArray(newShape, _workspace);
        RELEASE(newShape, _workspace);        
        
        if(rankOf() == copy.size() || copy.empty())
            reinterpret_cast<double *>(result->_buffer)[0] = NativeOpExcutioner::execSummaryStatsScalar(op, _buffer, _shapeInfo, nullptr, biasCorrected);
        else
            NativeOpExcutioner::execSummaryStats(op, _buffer, _shapeInfo, nullptr, result->_buffer, result->_shapeInfo, copy.data(), copy.size(), biasCorrected);

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
            return varianceAlongDimension(op, biasCorrected, std::vector<int>(dimensions));
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op, const NDArray *target, const bool biasCorrected, const std::vector<int>& dimensions) {
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        if(rankOf() == copy.size() || copy.empty())
            reinterpret_cast<double *>(target->_buffer)[0] = NativeOpExcutioner::execSummaryStatsScalar(op, _buffer, _shapeInfo, nullptr, biasCorrected);
        else
            NativeOpExcutioner::execSummaryStats(op, _buffer, _shapeInfo, nullptr, target->_buffer, target->_shapeInfo, copy.data(), copy.size(), biasCorrected);
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op,const NDArray *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension(op, target, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    NDArray NDArray::operator()(const std::vector<Nd4jLong>& idx, bool keepUnitiesInShape)  const {

        const int rank = rankOf();
        Nd4jLong *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(rank), Nd4jLong);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(rank));
        newShape[shape::shapeInfoLength(rank) - 2] = -1;

        auto shapeOf = shape::shapeOf(newShape);
        auto stridesOf = shape::stride(newShape);

        Nd4jLong offset = 0;
        Nd4jLong first, last;
        for (int d = 0; d < rank; ++d) {
            // building new shape first
            if (idx[2*d] != idx[2*d+1]) {

                first = idx[2*d]   >= 0 ? idx[2*d]   : idx[2*d]   + sizeAt(d) + 1;
                last  = idx[2*d+1] >= 0 ? idx[2*d+1] : idx[2*d+1] + sizeAt(d) + 1;

                shapeOf[d] = last - first;
                // for offset we're taking only the first index
                offset += first * stridesOf[d];
            }
        }

        NDArray result(_buffer + offset, newShape, _workspace);
        result._isShapeAlloc = true;

        if(!keepUnitiesInShape) {
            // check whether units are present in newShape, if yes then remove them by applying corresponding reshape
            // for example if result has shape {1,a,1,b} then after reshaping it acquire new shape {a,b}
            
            // std::vector<Nd4jLong > nonUnitDims;
            // for(int i = 0; i < result.rankOf(); ++i)
            //     if(newShape[i+1] != 1)
            //         nonUnitDims.push_back(newShape[i+1]);

            // if(nonUnitDims.size() != result.rankOf())
            //     result.reshapei(nonUnitDims);

            std::vector<Nd4jLong> nonUnitDims = ShapeUtils::evalDimsWithoutUnities(newShape);
            if(nonUnitDims.size() != result.rankOf())
                result.reshapei(nonUnitDims);
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape)  const {
        std::vector<Nd4jLong> idxRanges(2 * rankOf());
        
        ShapeUtils::evalIdxRangesForSubArr(subArrIdx, _shapeInfo, dimsToExclude, idxRanges.data());

        return (*this)(idxRanges, keepUnitiesInShape);
    }
      
    ////////////////////////////////////////////////////////////////////////
    // addition operator array + array
    NDArray NDArray::operator+(const NDArray& other) const {
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(this->_shapeInfo, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // addition operator array + scalar
    template <typename T>
    NDArray NDArray::operator+(const T scalar) const {

        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(this->_shapeInfo, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Add, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);

        return result;
    }
    template NDArray NDArray::operator+(const double scalar) const;
    template NDArray NDArray::operator+(const float scalar) const;
    template NDArray NDArray::operator+(const float16 scalar) const;
    template NDArray NDArray::operator+(const Nd4jLong scalar) const;
    template NDArray NDArray::operator+(const int scalar) const;
    template NDArray NDArray::operator+(const int16_t scalar) const;
    template NDArray NDArray::operator+(const int8_t scalar) const;
    template NDArray NDArray::operator+(const uint8_t scalar) const;
    template NDArray NDArray::operator+(const bool scalar) const;

    ////////////////////////////////////////////////////////////////////////
    // addition operator scalar + array
    // template<typename T>
    // NDArray<T> operator+(const T scalar, const NDArray<T>& arr) {
    //     return arr + scalar;
    // }
    ND4J_EXPORT NDArray operator+(const float16 scalar, const NDArray& arr) {
        return arr + scalar;        
    }

    ND4J_EXPORT NDArray operator+(const float scalar, const NDArray& arr) {
        return arr + scalar;        
    }

    ND4J_EXPORT NDArray operator+(const double scalar, const NDArray& arr) {
        return arr + scalar;        
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator scalar - array
    // template<typename T>
    // NDArray<T> operator-(const T scalar, const NDArray<T>& arr) {

    //     NDArray<T> result(arr._shapeInfo, false, arr._workspace);
    //     functions::scalar::ScalarTransform<T>::template transform<simdOps::ReverseSubtract<T>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

    //     return result;
    // }    
    ND4J_EXPORT NDArray operator-(const float16 scalar, const NDArray & arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;

    }

    ND4J_EXPORT NDArray operator-(const float scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ND4J_EXPORT NDArray operator-(const double scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ND4J_EXPORT NDArray operator-(const Nd4jLong scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ND4J_EXPORT NDArray operator*(const float16 scalar, const NDArray & arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;

    }

    ND4J_EXPORT NDArray operator*(const float scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ND4J_EXPORT NDArray operator*(const double scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ND4J_EXPORT NDArray operator*(const Nd4jLong scalar, const NDArray& arr) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(arr.getShapeInfo(), arr.getWorkspace());
        NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::operator+=(const NDArray& other) {
        if (!this->isScalar() && other.isScalar()) {
            NativeOpExcutioner::execScalar(nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
        } else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else{
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), &other, this, true);
            else
                *this = this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), other);

            RELEASE(bShape, this->_workspace);
        }
    }

    void NDArray::operator-=(const NDArray& other) {
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &other, this, true);
            else
                *this = this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), other);

            RELEASE(bShape, this->_workspace);
        }
    }

    template <typename T>
    void NDArray::operator+=(const T other) {
        auto tmp = NDArrayFactory::scalar(other);
        NativeOpExcutioner::execScalar(nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
    }
    template void NDArray::operator+=(const double other);
    template void NDArray::operator+=(const float other);
    template void NDArray::operator+=(const float16 other);
    template void NDArray::operator+=(const Nd4jLong other);
    template void NDArray::operator+=(const int other);
    template void NDArray::operator+=(const bool other);
    
    template<typename T>
    void NDArray::operator-=(const T other) {
        auto tmp = NDArrayFactory::scalar(other);
        NativeOpExcutioner::execScalar(nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
    }
    template void NDArray::operator-=(const double other);
    template void NDArray::operator-=(const float other);
    template void NDArray::operator-=(const float16 other);
    template void NDArray::operator-=(const Nd4jLong other);
    template void NDArray::operator-=(const int other);
    template void NDArray::operator-=(const bool other);


    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - array
    NDArray NDArray::operator-(const NDArray& other) const {
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, _workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), other);
    }


    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - scalar
    template<typename T>
    NDArray NDArray::operator-(const T& scalar) const {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(this->_shapeInfo, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);

        return result;
    }
    template NDArray NDArray::operator-(const double& scalar) const;
    template NDArray NDArray::operator-(const float& scalar) const;
    template NDArray NDArray::operator-(const float16& scalar) const;
    template NDArray NDArray::operator-(const Nd4jLong& scalar) const;
    template NDArray NDArray::operator-(const int& scalar) const;
    template NDArray NDArray::operator-(const bool& scalar) const;

    // negative operator, it makes all array elements = -elements
    NDArray NDArray::operator-() const {
        NDArray result(this->_shapeInfo, this->_workspace);
        NativeOpExcutioner::execTransform(nd4j::transform::Neg, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, nullptr, nullptr, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*array
    NDArray NDArray::operator*(const NDArray& other) const {
        
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(this->_shapeInfo, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), other);
    }


    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    NDArray NDArray::operator*(const T scalar) const {
        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(this->_shapeInfo, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);

        return result;
    }
    template NDArray NDArray::operator*(const double scalar) const;
    template NDArray NDArray::operator*(const float scalar) const;
    template NDArray NDArray::operator*(const float16 scalar) const;
    template NDArray NDArray::operator*(const Nd4jLong scalar) const;
    template NDArray NDArray::operator*(const int scalar) const;
    template NDArray NDArray::operator*(const uint8_t scalar) const;
    template NDArray NDArray::operator*(const int8_t scalar) const;
    template NDArray NDArray::operator*(const int16_t scalar) const;
    template NDArray NDArray::operator*(const bool scalar) const;

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array1 *= array2
    void NDArray::operator*=(const NDArray& other) {

        if (!this->isScalar() && other.isScalar()) {
            NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other.getBuffer(), other.getShapeInfo(), nullptr);
        } else if (other.lengthOf() == lengthOf())
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), &other, this, true);
            else
                *this = this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), other);

            RELEASE(bShape, this->_workspace);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    void NDArray::operator*=(const T scalar) {
        auto tmp = NDArrayFactory::scalar(scalar);

        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    }
    template void NDArray::operator*=(const double scalar);
    template void NDArray::operator*=(const float scalar);
    template void NDArray::operator*=(const float16 scalar);
    template void NDArray::operator*=(const Nd4jLong scalar);
    template void NDArray::operator*=(const int scalar);
    template void NDArray::operator*=(const int16_t scalar);
    template void NDArray::operator*=(const int8_t scalar);
    template void NDArray::operator*=(const uint8_t scalar);
    template void NDArray::operator*=(const bool scalar);


    ////////////////////////////////////////////////////////////////////////
    // division operator array/array
    NDArray NDArray::operator/(const NDArray& other) const {
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(this->_shapeInfo, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Divide, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array / scalar
    template<typename T>
    NDArray NDArray::operator/(const T scalar) const {
        if(scalar == (T)0.)
            throw std::runtime_error("NDArray::operator/ (division operator) : division by zero !");

        auto tmp = NDArrayFactory::scalar(scalar);

        NDArray result(this->_shapeInfo, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);

        return result;
    }
    template NDArray NDArray::operator/(const double scalar) const;
    template NDArray NDArray::operator/(const float scalar) const;
    template NDArray NDArray::operator/(const float16 scalar) const;
    template NDArray NDArray::operator/(const Nd4jLong scalar) const;
    template NDArray NDArray::operator/(const int scalar) const;
    template NDArray NDArray::operator/(const int8_t scalar) const;
    template NDArray NDArray::operator/(const uint8_t scalar) const;
    template NDArray NDArray::operator/(const int16_t scalar) const;
    template NDArray NDArray::operator/(const bool scalar) const;

    ////////////////////////////////////////////////////////////////////////
    // division operator array1 /= array2
    void NDArray::operator/=(const NDArray& other) {
        if (!this->isScalar() && other.isScalar()) {
            NativeOpExcutioner::execScalar(nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
        } else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Divide, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), &other, this, true);
            else
                *this = this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), other);

            RELEASE(bShape, this->_workspace);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array /= scalar
    template<typename T>
    void NDArray::operator/=(const T scalar) {
        auto tmp = NDArrayFactory::scalar(scalar);
        NativeOpExcutioner::execScalar(nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    }
    template void NDArray::operator/=(const double scalar);
    template void NDArray::operator/=(const float scalar);
    template void NDArray::operator/=(const float16 scalar);
    template void NDArray::operator/=(const Nd4jLong scalar);
    template void NDArray::operator/=(const int scalar);
    template void NDArray::operator/=(const int16_t scalar);
    template void NDArray::operator/=(const int8_t scalar);
    template void NDArray::operator/=(const uint8_t scalar);
    template void NDArray::operator/=(const bool scalar);

    ////////////////////////////////////////////////////////////////////////
    // mathematical multiplication of two arrays
    NDArray mmul(const NDArray& left, const NDArray& right) {
        auto ptr = MmulHelper::mmul(const_cast<NDArray*>(&left), const_cast<NDArray*>(&right), nullptr, 1., 0.);
        NDArray result(*ptr);
        delete ptr;
        return result;
    }

    DataType NDArray::dataType() const {
        return ArrayOptions::dataType(this->getShapeInfo());
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::assign(const NDArray& other, const Intervals& idx) {
        auto subarr = this->subarray(idx);
        subarr->assign(&other);
        delete subarr;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::setIdentity() {
        this->assign(0.);

        int  rank    = rankOf();
        auto shape   = shapeOf();
        auto strides = stridesOf();
        int  minDim  = 100000000;
        Nd4jLong indices[MAX_RANK];
        for(int j = 0; j < rank; ++j) 
                indices[j] = 1;
        
        Nd4jLong offset = shape::getOffset(0, shape, strides, indices, rank);
        
        for(int i = 0; i < rank; ++i) 
            if(minDim > shape[i])
                minDim = shape[i];

        float v = 1.0f;
#pragma omp parallel for if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i = 0; i < minDim; ++i)
            templatedSet<float>(_buffer, i*offset, this->dataType(), &v);
    }

    template <typename T>
    void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, void *value) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, void *value), LIBND4J_TYPES);



    template <typename T>
    void NDArray::templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
        auto x = reinterpret_cast<T *>(xBuffer);
        auto y = reinterpret_cast<T *>(yBuffer);

#pragma omp parallel for simd schedule(static)
        for (int i = 0; i < length; ++i) {
            auto temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSwap, (void *xBuffer, void *yBuffer, Nd4jLong length), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void NDArray::swapUnsafe(NDArray& other) {
        auto xType = this->dataType();

        if (xType != other.dataType())
            throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");
        
        if(_buffer == nullptr || other._buffer == nullptr)
            throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

        // if(_buffer == other._buffer)
        //     throw std::runtime_error("NDArray::swapUnsafe method: the buffers of input arrays should not point on the same address!");

        if(lengthOf() != other.lengthOf())
            throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

        BUILD_SINGLE_SELECTOR(xType, templatedSwap, (this->_buffer, other.buffer(), this->lengthOf()), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::diagonal(const char type) const {
        
        const char order = ordering();
        const int  rank  = rankOf();
        Nd4jLong *outShapeInfo;
        ALLOCATE(outShapeInfo, _workspace, 8, Nd4jLong);
        outShapeInfo[0] = 2;
        outShapeInfo[5] = 0;

        if(isVector() || isScalar()) {
            
            outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
            outShapeInfo[6] = 1;
            outShapeInfo[7] = (int)order;
        }
        else {            
            
            int diagSize  = 100000000;        
            Nd4jLong indices[MAX_RANK];
                    
            for(int i = 0; i < rank; ++i) {    
                if(diagSize > shapeOf()[i])
                    diagSize = shapeOf()[i];
                indices[i] = 1;
            }
            
            auto step = shape::getOffset(0, shapeOf(), stridesOf(), indices, rank);
                        
            if(type == 'c') {
                outShapeInfo[1] = diagSize;
                outShapeInfo[2] = 1;
            }
            else {
                outShapeInfo[1] = 1;
                outShapeInfo[2] = diagSize;                            
            }
            shape::updateStrides(outShapeInfo, order);
                        
            outShapeInfo[3] *= step;
            outShapeInfo[4] *= step;
            outShapeInfo[6] =  -1;
        }

        auto result = new NDArray(this->_buffer, outShapeInfo, this->_workspace);
        result->_isShapeAlloc = true;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
        if(rankOf() != 2)
           throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");

        const auto rows = sizeAt(0);
        const auto cols = sizeAt(1);
        
        switch(direction) {
            
            case 'u':                           // fill upper triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
                for(Nd4jLong i = 0; i < rows; ++i)
                    for(Nd4jLong j = 0; j < cols; ++j)
                        if (i + diag <= j)
                            p<T>(i, j, value);
                break;

            case 'l':                           // fill lower triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
                for(Nd4jLong i = 0; i < rows; ++i)
                    for(Nd4jLong j = 0; j < cols; ++j)
                        if (i + diag >= j)
                            p<T>(i, j, value);
                break;
            default:
                throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
        }
    }
    template void NDArray::setValueInDiagMatrix(const double& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const Nd4jLong& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int16_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const uint8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bool& value, const int diag, const char direction);

    ////////////////////////////////////////////////////////////////////////
    // default destructor
    NDArray::~NDArray() noexcept {
        if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr)
            delete[] _buffer;

        if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr)
            delete[] _shapeInfo;
    }
    
    void NDArray::streamline(char o) {
        char order = o == 'a' ? this->ordering() : o;
    
        Nd4jLong *newShape;
        ALLOCATE(newShape, this->_workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);

        int8_t *newBuffer;
        ALLOCATE(newBuffer, this->_workspace, this->lengthOf() * sizeOfT(), int8_t);

        std::vector<Nd4jLong> shape(this->rankOf());
        for (int e = 0; e < this->rankOf(); e++)
            shape[e] = this->sizeAt(e);

        if (order == 'c')
            shape::shapeBuffer(this->rankOf(),dataType(),  shape.data(), newShape);
        else
            shape::shapeBufferFortran(this->rankOf(), dataType(), shape.data(), newShape);

        if (!isView()) {
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Copy, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);
            memcpy(_buffer, newBuffer, this->lengthOf() * sizeOfT());

            //if (_isBuffAlloc)
            //    RELEASE(this->_buffer, this->_workspace);
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_workspace);

            //this->_buffer = newBuffer;
            //this->_isBuffAlloc = true;

            RELEASE(newBuffer, this->_workspace);

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        } else {
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Copy, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);

            if (_isBuffAlloc)
                RELEASE(this->_buffer, this->_workspace);
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_workspace);

            this->_buffer = newBuffer;
            this->_isBuffAlloc = true;

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        }
    }


    ////////////////////////////////////////////////////////////////////////
    void NDArray::tileToShape(const std::vector<Nd4jLong>& shape, NDArray* target) {
        if(target != nullptr) {
            this->tile(*target);
            return;
        }

        std::vector<Nd4jLong> thisShape(rankOf());
        for(int i = 0; i < rankOf(); ++i)
            thisShape[i] = sizeAt(i);

        if(!ShapeUtils::areShapesBroadcastable(shape, thisShape))
            throw std::runtime_error("NDArray::tileToShape method: the shape of this array and input shape are not suitable for broadcast operation !");

        const int newRank = shape.size();
        std::vector<Nd4jLong> repeats(newRank);

        for(int i = 1; i <= newRank; ++i) {
            if(i > rankOf())
                repeats[newRank-i] = shape[newRank - i];
            else
                repeats[newRank-i] = shape[newRank - i] / thisShape[rankOf() - i];
        }

        tilei(repeats);
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray* target) {
        const std::vector<Nd4jLong> shapeV(shape);
        tileToShape(shapeV, target);
    }

    ////////////////////////////////////////////////////////////////////////
    double NDArray::getTrace() const {
        int  rank    = rankOf();
        auto shape   = shapeOf();
        auto strides = stridesOf();
        int  minDim  = 100000000;
    
        Nd4jLong indices[MAX_RANK];
        for(int j = 0; j < rank; ++j)
            indices[j] = 1;
    
        auto offset = shape::getOffset(0, shape, strides, indices, rank);
    
        for(int i = 0; i < rank; ++i)
            if(minDim > shape[i])
                minDim = shape[i];

        double sum = 0.;

#pragma omp parallel for reduction(sumT:sum) if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i = 0; i < minDim; ++i)
            sum += _buffer[i*offset];

        return sum;
    }


////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::dupUninitialized() const {
        Nd4jLong* newShape(nullptr);
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(_shapeInfo), Nd4jLong);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(_shapeInfo));

        int8_t* buffer(nullptr);
        ALLOCATE(buffer, _workspace, _length * sizeOfT(), int8_t);
        auto result = new NDArray(buffer, newShape, _workspace);
        result->triggerAllocationFlag(true, true);

        return result;
    }

    NDArray NDArray::quantize(NDArray &array) {
        return *(quantize(&array));
    }

    NDArray* NDArray::quantize(NDArray *array) {
        auto ws = array->getWorkspace();
        char *buffer = nullptr;
        Nd4jLong *shapeInfo;

        // allocate buffers
        ALLOCATE(buffer, ws, TypeCast::estimateQuantizedSize(array->lengthOf()), char);
        COPY_SHAPE_EX(array->shapeInfo(), shapeInfo, ws);

        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_QUANTIZED);

        auto result = new NDArray(buffer, shapeInfo, ws);
        result->triggerAllocationFlag(true, true);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::multipleTensorsAlongDimension(const std::vector<int> &indices, const std::vector<int> &dimensions) const {
        auto result = new ResultSet();

        if (indices.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        auto tadLength = shape::tadLength(_shapeInfo, copy.data(), copy.size());
        auto numTads = _length / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        // FIXME: why we're not using workspaces here?
        Nd4jLong* shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("NDArray::multipleTensorsAlongDimension: index %i is higher then number of TADs: %i\n", idx, numTads);
                throw std::runtime_error("Bad index");
            }

            int8_t* buffer = _buffer + (tad->tadOffsets[idx] * sizeOfT());
            auto array = new NDArray(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allTensorsAlongDimension(const std::vector<int> &dimensions) const {
        auto result = new ResultSet();

        if(dimensions.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        if(copy.back() >= rankOf())
            throw std::runtime_error("NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");

        auto tadLength = shape::tadLength(_shapeInfo, copy.data(), copy.size());
        auto numTads = _length / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        auto shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (int idx = 0; idx < numTads; idx++ ) {
            auto buffer = _buffer + (tad->tadOffsets[idx] * sizeOfT());
            auto array = new NDArray(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allTensorsAlongDimension(const std::initializer_list<int>& dimensions) const {
        return allTensorsAlongDimension(std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allExamples() const {
        std::vector<int> dimensions(rankOf() - 1);
        for (int e = 1; e < rankOf(); e++)
            dimensions[e-1] = e;

        return allTensorsAlongDimension(dimensions);
    }

    //BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, Y value), LIBND4J_TYPES, LIBND4J_TYPES);
/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

