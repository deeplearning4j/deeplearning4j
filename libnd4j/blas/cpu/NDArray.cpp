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
    NDArray* NDArray::empty(nd4j::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace);
        result->_length = 0;
        result->triggerAllocationFlag(false, true);

        return result;
    }

    template <typename T>
    NDArray* NDArray::asT() {
        auto result = new NDArray(this->ordering(), this->getShapeAsVector());
        auto l = this->lengthOf();

        // FIXME: we want to avoid put/get indexed scalars here really
#pragma omp parallel for
        for (int e = 0; e < l; e++) {
            result->putIndexedScalar(e, this->getScalar(e));
        }

        return result;
    }

////////////////////////////////////////////////////////////////////////
// default constructor, do not allocate memory, memory for array is passed from outside
    NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, nd4j::memory::Workspace* workspace) {
        _buffer    = buffer;
        _shapeInfo = shapeInfo;
        _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
        _isShapeAlloc = false;
        _workspace = workspace;

        if (shapeInfo != nullptr)
            _length = shape::length(shapeInfo);
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
    template <typename T>
    NDArray* NDArray::create(std::initializer_list<Nd4jLong> s, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        std::vector<Nd4jLong> shape(s);
        int rank = (int) shape.size();


        ALLOCATE(res->_shapeInfo, workspace, shape::shapeInfoByteLength(rank), Nd4jLong);

        shape::shapeBuffer(rank, shape.data(), res->_shapeInfo);

        res->_length = shape::length(res->_shapeInfo);

        ALLOCATE(res->_buffer, workspace, res->_length * sizeof(T), int8_t);

        res->_isShapeAlloc = true;
        res->_isBuffAlloc = true;
        res->_workspace = workspace;

        return res;
    }

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArray::scalar(T scalar, nd4j::memory::Workspace* workspace) {
     auto res = new NDArray();

     res->_length = 1;
     ALLOCATE(res->_buffer, workspace, 1 * sizeof(T), int8_t);
     ALLOCATE(res->_shapeInfo, workspace, shape::shapeInfoByteLength(0), Nd4jLong);
     res->_shapeInfo[0] = 0;
     res->_shapeInfo[1] = 0;
     res->_shapeInfo[2] = 1;
     res->_shapeInfo[3] = 99;

     res->_isBuffAlloc = true;
     res->_isShapeAlloc = true;

     res->assign(scalar);

     return res;
}

#ifndef __JAVACPP_HACK__
    template <typename T>
    NDArray* NDArray::create(std::initializer_list<T> v, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        std::vector<T> values(v);
        ALLOCATE(res->_buffer, workspace, values.size() * sizeof(T), T);
        ALLOCATE(res->_shapeInfo, workspace, shape::shapeInfoByteLength(1), Nd4jLong);
        shape::shapeVector(values.size(), res->_shapeInfo);
        memcpy(res->_buffer, values.data(), values.size() * sizeof(T));

        res->_length = values.size();

        res->_isBuffAlloc = true;
        res->_isShapeAlloc = true;
        res->_workspace = workspace;

        return res;
    }

    template <typename T>
    NDArray* NDArray::create(std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        ALLOCATE(res->_buffer, workspace, values.size() * sizeof(T), int8_t);
        ALLOCATE(res->_shapeInfo, workspace, shape::shapeInfoByteLength(1), Nd4jLong);

        shape::shapeVector(values.size(), res->_shapeInfo);
        memcpy(res->_buffer, values.data(), values.size() * sizeof(T));

        res->_isBuffAlloc = true;
        res->_isShapeAlloc = true;
        res->_workspace = workspace;
        res->_length = values.size();
        return res;
    }
#endif

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
    NDArray::NDArray(const Nd4jLong* shapeInfo, const bool copyStrides, nd4j::memory::Workspace* workspace) {
   
    this->_length = shape::length(const_cast<Nd4jLong*>(shapeInfo));

    auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(shapeInfo));
    ALLOCATE(_buffer, workspace, this->_length * tLen, int8_t);
    ALLOCATE(_shapeInfo, workspace, shape::shapeInfoByteLength(shapeInfo), Nd4jLong);

    _workspace = workspace;

    memset(_buffer, 0, this->_length * tLen);          // set all elements in new array to be zeros

    memcpy(_shapeInfo, shapeInfo, shape::shapeInfoByteLength(shapeInfo));     // copy shape information into new array

    if(!copyStrides)
        shape::updateStrides(_shapeInfo, ordering());

    _isBuffAlloc = true;
    _isShapeAlloc = true;
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
            os << toStringValue(this->getIndexedScalar(e));

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
            vector[e] = this->getScalar<T>(e);
        }

        return vector;
    }

////////////////////////////////////////////////////////////////////////
    std::vector<Nd4jLong> NDArray::getShapeAsVector() {
        std::vector<Nd4jLong> vector(this->rankOf());

        for (int e = 0; e < this->rankOf(); e++)
            vector[e] = this->sizeAt(e);

        return vector;
    }

    std::vector<int64_t> NDArray::getShapeInfoAsFlatVector() {
        int magicNumber = shape::shapeInfoLength(this->rankOf());
        std::vector<int64_t> vector(magicNumber);

        for (int e = 0; e < magicNumber; e++)
            vector[e] = reinterpret_cast<int64_t>(this->_shapeInfo[e]);

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
/*
    template<typename T>
    void NDArray<T>::applyTriplewiseLambda(NDArray<T>* second, NDArray<T> *third, const std::function<T(T, T, T)>& func, NDArray<T>* target) {
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

    template<typename T>
    void NDArray<T>::applyPairwiseLambda(const NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target) {
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


////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::applyLambda(const std::function<T(T)>& func, NDArray<T>* target) {
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

    template<typename T>
    void NDArray<T>::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray<T>* target) {
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

                Nd4jLong xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                Nd4jLong zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func((Nd4jLong) e, this->_buffer[xOffset]);
            }
        }
    }

    template<typename T>
    void NDArray<T>::applyIndexedPairwiseLambda(NDArray<T>* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray<T>* target) {
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
    */
#endif

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {
    this->_length = shape::length(other->_shapeInfo);
    auto shapeLength = shape::shapeInfoByteLength(other->_shapeInfo);

    _workspace = workspace;
    auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(other->_shapeInfo));
    ALLOCATE(_buffer, workspace, this->_length * tLen, int8_t);
    ALLOCATE(_shapeInfo, workspace, shape::shapeInfoByteLength(other->_shapeInfo), Nd4jLong);
    // FIXME: memcpy should be removed
    // memcpy(_buffer, other->_buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

    memcpy(_shapeInfo, other->_shapeInfo, shapeLength);     // copy shape information into new array

    if(!copyStrides) 
        shape::updateStrides(_shapeInfo, ordering());

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

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

////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {

    this->_length = shape::length(other._shapeInfo);
    auto shapeLength = shape::shapeInfoByteLength(other._shapeInfo);

    _workspace = other._workspace;
    auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(other._shapeInfo));
    ALLOCATE(_buffer, other._workspace, this->_length * tLen, int8_t);
    ALLOCATE(_shapeInfo, other._workspace, shape::shapeInfoByteLength(other._shapeInfo), Nd4jLong);

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
        _bufferD = buffer;
        _shapeInfoD = shape;
    }

    nd4j::DataType NDArray::dataType() {
        return nd4j::ArrayOptions::dataType(this->shapeInfo());
    }

////////////////////////////////////////////////////////////////////////
// assignment operator
    NDArray& NDArray::operator=(const NDArray& other) {
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


void NDArray::replacePointers(void *buffer, Nd4jLong *shapeInfo, const bool releaseExisting ) {
    this->_buffer = buffer;
    this->_shapeInfo = shapeInfo;

    if (releaseExisting) {
        if (_isShapeAlloc && _workspace == nullptr)
            delete[] _shapeInfo;

        if (_isBuffAlloc && _workspace == nullptr)
            delete[] _buffer;
    }
}

    template<typename T>
    NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        Nd4jLong shapeOf[MAX_RANK];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        _workspace = workspace;
        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf);

            _buffer =  new T[shape::length(_shapeInfo)];
        } else {
            _buffer = reinterpret_cast<T*>(_workspace->allocateBytes(data.size() * sizeOfT()));
            _shapeInfo = reinterpret_cast<Nd4jLong*>(_workspace->allocateBytes(shape::shapeInfoByteLength(rank)));
            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, _shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, _shapeInfo);

            //_buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());

        }

        this->_length = shape::length(_shapeInfo);

        if (_length != data.size()) {
            nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(_shapeInfo));
            throw std::runtime_error("Data size doesn't match shape");
        }

        //memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        memcpy(_buffer, data.data(), sizeOfT() * this->_length);

		_isBuffAlloc = true;
		_isShapeAlloc = true;

        shape::updateStrides(_shapeInfo, order);
    }

    template<typename T>
    NDArray* NDArray::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        auto shapeOf = new Nd4jLong[rank];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        res->_workspace = workspace;
        if (workspace == nullptr) {
            if (order == 'f')
                res->_shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                res->_shapeInfo = shape::shapeBuffer(rank, shapeOf);

            res->_length = shape::length(res->_shapeInfo);
            res->_buffer =  new T[this->_length];
        } else {
            res->_shapeInfo = reinterpret_cast<Nd4jLong*>(res->_workspace->allocateBytes(shape::shapeInfoByteLength(rank)));

            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, res->_shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, res->_shapeInfo);

            res->_length = shape::length(res->_shapeInfo);

            res->_buffer = res->_workspace->allocateBytes(res->_length * res->sizeOfT());
        }

        memset(res->_buffer, 0, res->sizeOfT() * res->_length);
        
		res->_isBuffAlloc = true;
		res->_isShapeAlloc = true;

        shape::updateStrides(res->_shapeInfo, order);
        
        delete[] shapeOf;
        return res;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        auto shapeOf = new Nd4jLong[rank];
        int cnt = 0;

        for (const auto& item: shape)
            shapeOf[cnt++] = item;

        _workspace = workspace;
        _buffer = buffer;

        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf);
        } else {
            _shapeInfo = reinterpret_cast<Nd4jLong*>(_workspace->allocateBytes(shape::shapeInfoByteLength(rank)));

            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, _shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, _shapeInfo);
        }

        this->_length = shape::length(_shapeInfo);

        _isBuffAlloc = false;
        _isShapeAlloc = true;

        delete[] shapeOf;
    }

// This method assigns values of given NDArray to this one, wrt order
    void NDArray::assign(const NDArray *other) {
        if (this->isScalar() && other->isScalar()) {
            this ->_buffer[0] = other->_buffer[0];
            return;
        } else if (other->isScalar()) {
            this->assign(other->_buffer[0]);
            return;;
        }
        else if(other->isScalar()) {
            this->assign(other->_buffer[0]);
            return;
        }

        if (other->lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other->ordering() && shape::elementWiseStride(this->_shapeInfo) == 1 && shape::elementWiseStride(other->_shapeInfo) == 1) {
            memcpy(_buffer, other->_buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner::execPairwiseTransform(1, _buffer, _shapeInfo, other->_buffer, other->_shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }

// This method assigns values of given NDArray to this one
    void NDArray::assign(const NDArray& other) {
        if (this->isScalar() && other.isScalar()) {
            this->_buffer[0] = other._buffer[0];
            return;
        } else if (other.isScalar()) {
            this->assign(other._buffer[0]);
            return;;
        }

        if (this == &other) 
            return;
        if (other.lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && shape::elementWiseStride(_shapeInfo) == 1 && shape::elementWiseStride(other._shapeInfo) == 1) {
            
            memcpy(_buffer, other._buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner::execPairwiseTransform(1, _buffer, _shapeInfo, other._buffer, other._shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }

// This method assigns given value to all elements in this NDArray
    template<typename T>
    void NDArray::assign(const T value) {
        // just fire scalar
        NativeOpExcutioner::execScalar(13, _buffer, _shapeInfo, _buffer, _shapeInfo, value, nullptr);
    }

    NDArray* NDArray::detach() {
        if (!isAttached())
            return this;

        Nd4jLong newLength = shape::length(_shapeInfo);
        void* newBuffer;
        Nd4jLong* newShapeInfo;

        newBuffer = new int8_t[newLength * sizeOfT()];

        if (this->ordering() == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());

        auto result = new NDArray(newBuffer, newShapeInfo, nullptr);
        result->_isBuffAlloc = true;
        result->_isShapeAlloc = true;

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
            newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());

    } else {
        newBuffer = _workspace->allocateBytes(newLength * sizeOfT());
        newShapeInfo = reinterpret_cast<Nd4jLong *>(_workspace->allocateBytes(shape::shapeInfoByteLength(this->rankOf())));

        if (order == 'f')
            shape::shapeBufferFortran(rankOf(), shapeOf(), newShapeInfo);
        else
            shape::shapeBuffer(rankOf(), shapeOf(), newShapeInfo);
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
        // FIXME: leak
        auto res = NDArray::scalar(this->dataType(), 0.0, this->_workspace);
        NativeOpExcutioner::execSummaryStats(op,this->getBuffer(), this->getShapeInfo(), nullptr, res->buffer(), res->shapeInfo(), biasCorrected);
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    NDArray NDArray::sumNumber() const {
        // FIXME: leak
        auto res = NDArray::scalar(this->dataType(), 0.0, this->_workspace);
        NativeOpExcutioner::execReduceScalar(1, _buffer, _shapeInfo, nullptr, res->buffer(), res->shapeInfo());
        return *res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    NDArray NDArray::meanNumber() const {
        // FIXME: leak
        auto res = NDArray::scalar(this->dataType(), 0.0, this->_workspace);
        NativeOpExcutioner::execReduceScalar(0, _buffer, _shapeInfo, nullptr, res->buffer(), res->shapeInfo());
        return *res;
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
        return static_cast<int>(this->template reduceNumber<simdOps::IsNan<T>>()) > 0;
    }

    bool NDArray::hasInfs() {
        return static_cast<int>(this->template reduceNumber<simdOps::IsInf<T>>()) > 0;
    }

    bool NDArray::isFinite() {
        return static_cast<int>(this->template reduceNumber<simdOps::IsInfOrNan<T>>()) == 0;
    }

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

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, result->getBuffer(), result->getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets)
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

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, result.getBuffer(), result.getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets)
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

            NativeOpExcutioner::execReduce(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets)
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
            printf("%f", this->getIndexedScalar(e));
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

        T* buffer = this->_buffer + tad.tadOffsets[index];

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

    // This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
    bool NDArray::equalsTo(const NDArray *other, double eps) const {

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

        T *extras = new T[1]{eps};

        // we don't need extraparams for this op
        double val = NativeOpExcutioner::execReduce3Scalar(4, _buffer, _shapeInfo, extras, other->_buffer, other->_shapeInfo);

        delete[] extras;

        if (val > 0)
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

    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::Ops op, T scalar, NDArray *target, void *extraParams) const {
        auto temp = NDArray::scalar<T>(scalar, this->_workspace);

        if (target == nullptr)
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, temp->buffer(), temp->shapeInfo(), extraParams);
        else
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, temp->buffer(), temp->shapeInfo(), extraParams);

        delete temp;
    }

    void NDArray::applyScalar(nd4j::scalar::Ops op, NDArray& scalar, NDArray* target, void *extraParams) const {
        if (!scalar.isScalar()) {
            throw std::runtime_error("Operand is not a scalar!");
        }

        if (target == nullptr)
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar.buffer(), scalar.shapeInfo(), extraParams);
        else
            NativeOpExcutioner::execScalar(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, scalar.buffer(), scalar.shapeInfo(), extraParams);
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
            shape::shapeBuffer(dimensions.size(), dimensions.data(), newShape);
        else
            shape::shapeBufferFortran(dimensions.size(), dimensions.data(), newShape);

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
            shape::shapeBuffer(shape.size(), shape.data(), shapeInfoNew);
        else
            shape::shapeBufferFortran(shape.size(), shape.data(), shapeInfoNew);

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
            T mv = -MAX_FLOAT;
            for (Nd4jLong e = 0; e < this->lengthOf(); e++) {
                T val = this->getScalar(e);
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
    ALLOCATE(newBuff, _workspace, shape::length(newShapeInfo), int8_t);
    // assign new shape and new buffer to resulting array
    NDArray result(newBuff, newShapeInfo, _workspace);
    result._isShapeAlloc = true;
    result._isBuffAlloc = true;

    // fill newBuff, loop through all elements of newBuff 
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const auto resultLen = result.lengthOf();
    if(result.ordering() == 'c') {           //  ews == 1 always here
#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<resultLen; ++i)
            newBuff[i] = (*this)(shape::subArrayIndex(newShapeInfo, _shapeInfo, i));
    }
    else {
        Nd4jLong idx[MAX_RANK];
        auto resultShape   = result.shapeOf();
        auto resultStrides = result.stridesOf();
        const auto resultRank = result.rankOf();
#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(int i=0;  i<resultLen; ++i) {
            shape::ind2subC(resultRank, resultShape, i, resultLen, idx);
            newBuff[ shape::getOffset(0, resultShape, resultStrides, idx, resultRank) ] = (*this)(shape::subArrayIndex(newShapeInfo, _shapeInfo, i));
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
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(Nd4jLong i=0;  i<targetLen; ++i)
                targetBuff[i] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
        else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i)
                targetBuff[i*ews] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
        else {
            Nd4jLong idx[MAX_RANK];
            auto targetShape     = target.shapeOf();
            auto targetStrides   = target.stridesOf();
            const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
            for(int i=0;  i<targetLen; ++i) {
                shape::ind2subC(targetRank, targetShape, i, targetLen, idx);
                targetBuff[ shape::getOffset(0, targetShape, targetStrides, idx, targetRank) ] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
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
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i)
                targetBuff[i] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
        else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i)
                targetBuff[i*ews] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
        else {
            Nd4jLong idx[MAX_RANK];
            auto targetShape     = target.shapeOf();
            auto targetStrides   = target.stridesOf();
            const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
            for(int i=0;  i<targetLen; ++i) {
                shape::ind2subC(targetRank, targetShape, i, targetLen, idx);
                targetBuff[ shape::getOffset(0, targetShape, targetStrides, idx, targetRank) ] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
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
        auto outShape = ShapeUtils<T>::evalRepeatShape(dimension, repeats, *this);
    
        // the size of outShape == rank
        int rank = rankOf();            // = outShape.size()

        auto newShape = new Nd4jLong[rank];
        for (int i = 0; i < rank; i++)
            newShape[i] = outShape[i];

        auto ret = new NDArray('c', outShape, _workspace);

        auto repeatDelta = shape::prodLong(newShape, rank) / this->lengthOf();
        auto numTads = this->tensorsAlongDimension({dimension});
        for (int i = 0; i < numTads; i++) {
            auto thisTensor = this->tensorAlongDimension(i, {dimension});
            auto retTensor = ret->tensorAlongDimension(i, {dimension});
            int retIdx = 0;
            for (int k = 0; k < thisTensor->lengthOf(); k++) {
                T s = thisTensor->getIndexedScalar(k);
                for (int j = 0; j < repeatDelta; j++) {
                    retTensor->putIndexedScalar(retIdx++, s);
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
            for (int k = 0; k < thisTensor->lengthOf(); k++) {
                T s = thisTensor->getIndexedScalar(k);
                for (int j = 0; j < repeatDelta; j++) {
                    retTensor->putIndexedScalar(retIdx++, s);
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
    void NDArray::applyTrueBroadcast(nd4j::broadcast::Ops op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast method: target or other = nullptr !");

        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op, const_cast<NDArray*>(other), extraArgs);
            return;
        }
    if (other->isScalar()) {        
        this->template applyScalar<OpName>(other->getScalar(0), target, extraArgs);
        return;
    }

    const NDArray<T>* min(nullptr), *max(nullptr);
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
        if(!ShapeUtils<T>::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
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

    NDArray<T>* pMin = const_cast<NDArray<T>*>(min);
    if(product != 1 )
        pMin = new NDArray<T>(min->tile(repeatMin));

    std::vector<int> sameDims = ShapeUtils<T>::getDimsWithSameShape(target, pMin);

    if(max == this) {        
        target->template applyBroadcast<OpName>(sameDims, pMin, nullptr, extraArgs);
    }
    else {
    
        std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(target->rankOf(), sameDims);
        const Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);
        
#pragma omp parallel for schedule(guided)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
            
            NDArray<T> targetSubArr = (*target)(i, dimsToExclude);
            pMin->template applyPairwiseTransform<OpName>(&targetSubArr, &targetSubArr, extraArgs);
        }
    }

    if(pMin != min)
        delete pMin;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T>* NDArray<T>::applyTrueBroadcast(const NDArray<T>* other, T *extraArgs) const {

    Nd4jLong* newShapeInfo = nullptr;
    if(!ShapeUtils<T>::evalBroadcastShapeInfo(*this, *other, true, newShapeInfo, _workspace))          // the rank of new array = max->rankOf)()
        throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
    auto result = new NDArray<T>(newShapeInfo, false, this->_workspace);

    // if workspace is not null - do not call delete.
    if (_workspace == nullptr)
        delete[] newShapeInfo;

    this->template applyTrueBroadcast<OpName>(other, result, false, extraArgs);
  
    return result;
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T> NDArray<T>::applyTrueBroadcast(const NDArray<T>& other, T *extraArgs) const {

    NDArray<T>* pResult = this->template applyTrueBroadcast<OpName>(&other, extraArgs);
    pResult->_isShapeAlloc = false;
    pResult->_isBuffAlloc  = false;

    NDArray<T> result(pResult->_buffer, pResult->_shapeInfo, _workspace);
    result._isShapeAlloc = true;
    result._isBuffAlloc  = true;
    
    delete pResult;

    return result;
}


//////////////////////////////////////////////////////////////////////////
// return array which is broadcasted from this and argument array  
template<typename T>
NDArray<T>* NDArray<T>::broadcast(const NDArray<T>& other) {	
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
		biggerRank = _shapeInfo[0];
		smallerShapeInfo = other._shapeInfo;
		smallerRank = other._shapeInfo[0];
	}
	else {
		biggerShapeInfo = other._shapeInfo;
		biggerRank = other._shapeInfo[0];
		smallerShapeInfo = _shapeInfo;
		smallerRank = _shapeInfo[0];
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

	NDArray<T>* ret = new NDArray<T>(shapeInfoNew, _workspace);
	ret->updateStrides(order);
	delete []shapeInfoNew;

	return ret;
}


//////////////////////////////////////////////////////////////////////////
// check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
template<typename T>
bool NDArray<T>::hasOrthonormalBasis(const int arg) {

	if(rankOf() !=2 )
		throw std::runtime_error("NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !");

	if(arg!=0  && arg!=1)
		throw std::runtime_error("NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !");

	const T eps = 1e-5f;
	T dot = 0.f;
	if(arg) {					// check whether columns create orthogonal basis
		for(int j=0; j<columns()-1; ++j)
			for(int k=j+1; k<columns(); ++k) {
				for(int i=0; i<rows(); ++i)
					dot += getScalar(i,j)*getScalar(i,k);
				if(nd4j::math::nd4j_abs(dot) > eps )
					return false;
				dot = 0.f;
			}
		for(int j=0; j<columns(); ++j)	{	// check whether norm of column vector = 1
			for(int i=0; i<rows(); ++i)
				dot += getScalar(i,j)*getScalar(i,j);
			if(dot != (T) 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - (T) 1.f) > eps)
				return false;
			dot = 0.f;
		}
	}
	else {						// check whether rows create orthogonal basis
		for(int i=0; i<rows()-1; ++i)
			for(int k=i+1; k<rows(); ++k) {
				for(int j=0; j<columns(); ++j)
					dot += getScalar(i,j)*getScalar(k,j);
				if(nd4j::math::nd4j_abs(dot) > eps )
					return false;
				dot = (T) 0.f;
			}
		for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			for(int j=0; j<columns(); ++j)
					dot += getScalar(i,j)*getScalar(i,j);
			if(dot!= (T) 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - (T) 1.f) > eps)
				return false;
			dot = 0.f;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is identity matrix
template<typename T>
bool NDArray<T>::isIdentityMatrix() {
	if(rankOf() !=2 || rows() != columns())
		throw std::runtime_error("isIdentityMatrix method: matrix must be square and have rank = 2 !");

	const T eps = 1e-5f;
	for(int i=0; i<rows(); ++i)
		if(nd4j::math::nd4j_abs(getScalar(i,i) - 1.f) > eps)
			return false;

	for(int i=0; i<rows(); ++i)
		for(int j=0; j<columns(); ++j) {
			if (i == j) continue;
			if(nd4j::math::nd4j_abs(getScalar(i,j)) > eps)
				return false;
		}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is unitary matrix
template<typename T>
bool NDArray<T>::isUnitary() {

    if(rankOf() !=2 || rows() != columns())
        throw std::runtime_error("isUnitary method: matrix must be square and have rank = 2 !");

    NDArray<T>* tr = this->transpose();
    NDArray<T>* trMul = MmulHelper<T>::mmul(this, tr, nullptr, 1.f, 0.f);

    bool result = trMul->isIdentityMatrix();
    delete tr;
    delete trMul;
    
    return result;
}

////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(IndicesList& idx, std::vector<Nd4jLong>& strides) const {
        auto raw = subarray(idx);

        for (int e = 0; e < strides.size(); e++)
            raw->stridesOf()[e] *= strides[e];

        return raw;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(IndicesList& idx) const {
        // scalar subarray edge case
        if (idx.isScalar()) {
            auto pnt = idx.at(0)->getIndices().at(0);
            return new NDArray<T>('c', {1, 1}, {this->getScalar(pnt)});   
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

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(const std::initializer_list<NDIndex*>& idx) const {
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

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        for (auto v: idx) {
            delete v;
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(const Intervals& idx) const {

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

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }

    template <typename T>
    NDArray<T>* NDArray<T>::cast(DataType dtype) {
        // TODO: to be implemented
        return nullptr;
    }

    template <typename T>
    void NDArray<T>::cast(NDArray<T>* target, DataType dtype) {
        // TODO: to be implemented properly
        target->assign(this);
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyIndexReduce(const NDArray<T>* target, const std::vector<int>& dimensions, const T *extraParams) const {
        if (target->isScalar()) {
            target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
        } else {
            std::vector<int> copy(dimensions);
            if (dimensions.size() > 1)
                std::sort(copy.begin(), copy.end());

            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            functions::indexreduce::IndexReduce<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), target->_buffer,
                                                                          target->_shapeInfo, copy.data(), copy.size(),
                                                                          tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
    }
    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyIndexReduce(const std::vector<int>& dimensions, const T* extraParams ) const {
        
        std::vector<int> copy(dimensions);
        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size())
            result->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
        
            functions::indexreduce::IndexReduce<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), result->_buffer,
                                                                    result->_shapeInfo, copy.data(), copy.size(),
                                                                    tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const T* extraParams) const {
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
        auto result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        
        result->_buffer[0] = functions::reduce3::Reduce3<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), other->_buffer, other->_shapeInfo);        
        
        delete []extraParamsVals;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>*  NDArray<T>::applyAllReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {
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
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        functions::reduce3::Reduce3<T>::template execAll<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size(), tadX.tadOnlyShapeInfo, tadX.tadOffsets, tadY.tadOnlyShapeInfo, tadY.tadOffsets);
        delete []extraParamsVals;
        return result;
    }
 
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {
        
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               

        auto newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size())
            result->_buffer[0] = functions::reduce3::Reduce3<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), other->_buffer, other->_shapeInfo);
        else {
            shape::TAD tadX(_shapeInfo, copy.data(), copy.size());
            tadX.createTadOnlyShapeInfo();
            tadX.createOffsets();

            shape::TAD tadY(other->_shapeInfo, copy.data(), copy.size());
            tadY.createTadOnlyShapeInfo();
            tadY.createOffsets();        
        
            functions::reduce3::Reduce3<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size(), tadX.tadOnlyShapeInfo, tadX.tadOffsets);
        }
        
        delete []extraParamsVals;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::vector<int>& dimensions) const {
    
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());
            
        auto newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);        
        
        if(rankOf() == copy.size() || copy.empty())
            result->_buffer[0] = functions::summarystats::SummaryStatsReduce<T>::template execScalar<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr);
        else
            functions::summarystats::SummaryStatsReduce<T>::template exec<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr,
                                                                              result->_buffer, result->_shapeInfo, copy.data(), copy.size());
        return result;    
    
    }
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
    
        return varianceAlongDimension<OpName>(biasCorrected, std::vector<int>(dimensions));
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::vector<int>& dimensions) {
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        if(rankOf() == copy.size() || copy.empty())
            target->_buffer[0] = functions::summarystats::SummaryStatsReduce<T>::template execScalar<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr);
        else
            functions::summarystats::SummaryStatsReduce<T>::template exec<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr,
                                                                                  target->_buffer, target->_shapeInfo, copy.data(), copy.size());

    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension<OpName>(target, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    template<typename T>
    NDArray<T> NDArray<T>::operator()(const std::vector<Nd4jLong>& idx, bool keepUnitiesInShape)  const {

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

        NDArray<T> result(_buffer + offset, newShape, _workspace);
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

            std::vector<Nd4jLong> nonUnitDims = ShapeUtils<T>::evalDimsWithoutUnities(newShape);
            if(nonUnitDims.size() != result.rankOf())
                result.reshapei(nonUnitDims);
        }

        return result;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T> NDArray<T>::operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape)  const {

    std::vector<Nd4jLong> idxRanges(2 * rankOf());        
        
    ShapeUtils<T>::evalIdxRangesForSubArr(subArrIdx, _shapeInfo, dimsToExclude, idxRanges.data());

    return (*this)(idxRanges, keepUnitiesInShape);
}
      
////////////////////////////////////////////////////////////////////////
// addition operator array + array
template<typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& other) const {

    if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NDArray<T> result(this->_shapeInfo, this->_workspace);        
        functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
        return result;
    }

    return this->template applyTrueBroadcast<simdOps::Add<T>>(other);

}

    ////////////////////////////////////////////////////////////////////////
    // addition operator array + scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator+(const T scalar) const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // addition operator scalar + array
    // template<typename T>
    // NDArray<T> operator+(const T scalar, const NDArray<T>& arr) {
    //     return arr + scalar;
    // }
    ND4J_EXPORT NDArray<float16> operator+(const float16 scalar, const NDArray<float16>& arr) {
        return arr + scalar;        
    }
    ND4J_EXPORT NDArray<float> operator+(const float scalar, const NDArray<float>& arr) {
        return arr + scalar;        
    }
    ND4J_EXPORT NDArray<double> operator+(const double scalar, const NDArray<double>& arr) {
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
    ND4J_EXPORT NDArray<float16> operator-(const float16 scalar, const NDArray<float16>& arr) {
        
        NDArray<float16> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<float16>::template transform<simdOps::ReverseSubtract<float16>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }        
    ND4J_EXPORT NDArray<float> operator-(const float scalar, const NDArray<float>& arr) {
        
        NDArray<float> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<float>::template transform<simdOps::ReverseSubtract<float>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }        
    ND4J_EXPORT NDArray<double> operator-(const double scalar, const NDArray<double>& arr) {
        
        NDArray<double> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<double>::template transform<simdOps::ReverseSubtract<double>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }    
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::operator+=(const NDArray<T>& other) {    
        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else{
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils<T>::evalBroadcastShapeInfo(this, &other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->template applyTrueBroadcast<simdOps::Add<T>>(&other, this, true);
            else
                *this = this->template applyTrueBroadcast<simdOps::Add<T>>(other);

            RELEASE(bShape, this->_workspace);
        }
    }

    template<typename T>
    void NDArray<T>::operator-=(const NDArray<T>& other) {    

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils<T>::evalBroadcastShapeInfo(this, &other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->template applyTrueBroadcast<simdOps::Subtract<T>>(&other, this, true);
            else
                *this = this->template applyTrueBroadcast<simdOps::Subtract<T>>(other);

            RELEASE(bShape, this->_workspace);
        }
    }

    template<typename T>
    void NDArray<T>::operator+=(const T other) {
        if (this->isScalar())
            this->_buffer[0] += other;
        else
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other, nullptr);
    }
    
    template<typename T>
    void NDArray<T>::operator-=(const T other) {  
        if (this->isScalar())
            this->_buffer[0] -= other;
        else  
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other, nullptr);
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - array
    template<typename T>
    NDArray<T> NDArray<T>::operator-(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray<T> result(_shapeInfo, _workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Subtract<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator-(const T& scalar) const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    // negative operator, it makes all array elements = -elements
    template<typename T>
    NDArray<T> NDArray<T>::operator-() const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::transform::Transform<T>::template exec<simdOps::Neg<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, nullptr, nullptr, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*array
    template<typename T>
    NDArray<T> NDArray<T>::operator*(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray<T> result(this->_shapeInfo, this->_workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Multiply<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator*(const T scalar) const {
        
        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array1 *= array2
    template<typename T>
    void NDArray<T>::operator*=(const NDArray<T>& other) {    

        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils<T>::evalBroadcastShapeInfo(this, &other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->template applyTrueBroadcast<simdOps::Multiply<T>>(&other, this, true);
            else
                *this = this->template applyTrueBroadcast<simdOps::Multiply<T>>(other);

            RELEASE(bShape, this->_workspace);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    void NDArray<T>::operator*=(const T scalar) {
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, nullptr);
    }


    ////////////////////////////////////////////////////////////////////////
    // division operator array/array
    template<typename T>
    NDArray<T> NDArray<T>::operator/(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray<T> result(this->_shapeInfo, this->_workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Divide<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array / scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator/(const T scalar) const {

        if(scalar == (T)0.)
            throw std::runtime_error("NDArray::operator/ (division operator) : division by zero !");
        
        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array1 /= array2
    template<typename T>
    void NDArray<T>::operator/=(const NDArray<T>& other) {    

        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else {
            Nd4jLong *bShape = nullptr;
            auto p = ShapeUtils<T>::evalBroadcastShapeInfo(this, &other, true, bShape, this->_workspace);
            if (shape::shapeEquals(this->shapeInfo(), bShape))
                this->template applyTrueBroadcast<simdOps::Divide<T>>(&other, this, true);
            else
                *this = this->template applyTrueBroadcast<simdOps::Divide<T>>(other);

            RELEASE(bShape, this->_workspace);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array /= scalar
    template<typename T>
    void NDArray<T>::operator/=(const T scalar) {
        
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, nullptr);
    }

    ////////////////////////////////////////////////////////////////////////
    // mathematical multiplication of two arrays
    template<typename T>
    NDArray<T> mmul(const NDArray<T>& left, const NDArray<T>& right) {

        NDArray<T>* ptr =  MmulHelper<T>::mmul(const_cast<NDArray<T>*>(&left), const_cast<NDArray<T>*>(&right), nullptr, (T)1., (T)0.);
        NDArray<T> result(*ptr);
        delete ptr;
        return result;
    }

    template<typename T>
    DataType NDArray<T>::dataType() const {
        return _dataType;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::assign(const NDArray<T>& other, const Intervals& idx) {

        NDArray<T>* subarr = this->subarray(idx);
        subarr->assign(&other);
        delete subarr;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::setIdentity() {

        this->assign((T)0.);

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

#pragma omp parallel for if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i = 0; i < minDim; ++i)
            _buffer[i*offset] = (T)1.;                
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::swapUnsafe(NDArray<T>& other) {
        
        if(_buffer == nullptr || other._buffer == nullptr)
            throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

        // if(_buffer == other._buffer)
        //     throw std::runtime_error("NDArray::swapUnsafe method: the buffers of input arrays should not point on the same address!");

        if(lengthOf() != other.lengthOf())
            throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

        T temp;
#pragma omp parallel for schedule(static) private(temp)
        for (int i = 0; i < lengthOf(); ++i) {
            temp = (*this)(i);
            (*this)(i) = other(i);
            other(i) = temp;            
        }
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::diagonal(const char type) const {        
        
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

        NDArray<T>* result = new NDArray<T>(this->_buffer, outShapeInfo, this->_workspace);
        result->_isShapeAlloc = true;
        return result;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::setValueInDiagMatrix(const T& value, const int diag, const char direction) {

    if(rankOf() != 2)
       throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");

    const int rows = sizeAt(0);
    const int cols = sizeAt(1);
        
    switch(direction) {
            
        case 'u':                           // fill upper triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
            for(int i = 0; i < rows; ++i) 
                for(int j = 0; j < cols; ++j)                                      
                    if (i + diag <= j)
                        (*this)(i, j) = value;    
                break;

        case 'l':                           // fill lower triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
            for(int i = 0; i < rows; ++i) 
                for(int j = 0; j < cols; ++j)                                      
                    if (i + diag >= j)
                        (*this)(i, j) = value;    
            break;

        default:
            throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
    }  
}

    ////////////////////////////////////////////////////////////////////////
    // default destructor
    template<typename T>
    NDArray<T>::~NDArray() noexcept {
        if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr)
            delete[] _buffer;

        if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr)
            delete[] _shapeInfo;
    }
    
template<typename T>
void NDArray<T>::streamline(char o) {
    char order = o == 'a' ? this->ordering() : o;
    
    Nd4jLong *newShape;
    ALLOCATE(newShape, this->_workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);

    T *newBuffer;
    ALLOCATE(newBuffer, this->_workspace, this->lengthOf(), T);

    std::vector<Nd4jLong> shape(this->rankOf());
    for (int e = 0; e < this->rankOf(); e++)
        shape[e] = this->sizeAt(e);

    if (order == 'c')
        shape::shapeBuffer(this->rankOf(), shape.data(), newShape);
    else
        shape::shapeBufferFortran(this->rankOf(), shape.data(), newShape);

    if (!isView()) {

        NativeOpExcutioner<T>::execPairwiseTransform(1, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);
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
        NativeOpExcutioner<T>::execPairwiseTransform(1, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);

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
template<typename T>
void NDArray<T>::tileToShape(const std::vector<Nd4jLong>& shape, NDArray<T>* target) {

    if(target != nullptr) {
        this->tile(*target);
        return;
    }

    std::vector<Nd4jLong> thisShape(rankOf());
    for(int i = 0; i < rankOf(); ++i)
        thisShape[i] = sizeAt(i);

    if(!ShapeUtils<T>::areShapesBroadcastable(shape, thisShape))
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
template<typename T>
void NDArray<T>::tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray<T>* target) {

    const std::vector<Nd4jLong> shapeV(shape);
    tileToShape(shapeV, target);
}

////////////////////////////////////////////////////////////////////////
template<typename T>
T NDArray<T>::getTrace() const {
    
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
    T sum = 0.;

#pragma omp parallel for reduction(sumT:sum) if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
    for(int i = 0; i < minDim; ++i)
        sum += _buffer[i*offset];

    return sum;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order) {
    
    NDArray<T>* result = new NDArray<T>(order, shape);
    result->assign(value);
    return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order) {

    return valueOf(std::vector<Nd4jLong>(shape), value, order);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::linspace(const T from, const T to, const Nd4jLong numElements) {

    NDArray<T>* result = new NDArray<T>('c', {1, (int)numElements});

    for (Nd4jLong e = 0; e < numElements; e++) {
        T step = (T) e / ((T) numElements - (T) 1.0f);
        result->getBuffer()[e] = (from * ((T) 1.0f - step) + step * to);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::linspace(const T from, const T step) {
            
    for (Nd4jLong i = 0; i < _length; ++i)
        (*this)(i) = from + (step * i);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::scalar(const T value) {

    NDArray<T>* result = new NDArray<T>('c', {1, 1});
    result->putScalar(0, value);

    return result;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::createUninitialized() const {
    
    Nd4jLong* newShape(nullptr);
    ALLOCATE(newShape, _workspace, shape::shapeInfoLength(_shapeInfo), Nd4jLong);
    memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(_shapeInfo));

    T* buffer(nullptr);
    ALLOCATE(buffer, _workspace, _length, T);
    NDArray<T>* result = new NDArray<T>(buffer, newShape, _workspace);
    result->triggerAllocationFlag(true, true);

    return result;
}

template<typename T>
NDArray<T> NDArray<T>::quantize(NDArray<T> &array) {
    return *(quantize(&array));
}

template<typename T>
NDArray<T>* NDArray<T>::quantize(NDArray<T> *array) {
    auto ws = array->getWorkspace();
    char *buffer = nullptr;
    Nd4jLong *shapeInfo;

    // allocate buffers
    ALLOCATE(buffer, ws, TypeCast::estimateQuantizedSize(array->lengthOf()), char);
    COPY_SHAPE_EX(array->shapeInfo(), shapeInfo, ws);

    ArrayOptions::setPropertyBit(shapeInfo, ARRAY_QUANTIZED);

    auto result = new NDArray<T>(reinterpret_cast<T *>(buffer), shapeInfo, ws);
    result->triggerAllocationFlag(true, true);

    return result;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
ResultSet<T>* NDArray<T>::multipleTensorsAlongDimension(const std::vector<int> &indices, const std::vector<int> &dimensions) const {
    
    ResultSet<T>* result = new ResultSet<T>();

    if (indices.size() == 0)
        return result;

    std::vector<int> copy(dimensions);

    // we need to sort dimensions (?)
    if (dimensions.size() > 1)
        std::sort (copy.begin(), copy.end());

    Nd4jLong tadLength = shape::tadLength(_shapeInfo, copy.data(), copy.size());
    Nd4jLong numTads = _length / tadLength;

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

        T* buffer = _buffer + tad->tadOffsets[idx];
        NDArray<T>* array = new NDArray<T>(buffer, shapeInfo);
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
template<typename T>
ResultSet<T>* NDArray<T>::allTensorsAlongDimension(const std::vector<int> &dimensions) const {
        
    ResultSet<T>* result = new ResultSet<T>();

    if(dimensions.size() == 0)
        return result;

    std::vector<int> copy(dimensions);

    // we need to sort dimensions (?)
    if (dimensions.size() > 1)
        std::sort (copy.begin(), copy.end());

    if(copy.back() >= rankOf())
        throw std::runtime_error("NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");

    Nd4jLong tadLength = shape::tadLength(_shapeInfo, copy.data(), copy.size());
    Nd4jLong numTads = _length / tadLength;

    std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, copy.data(), copy.size()));
    tad->createTadOnlyShapeInfo();
    tad->createOffsets();

    auto shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
    std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

    for (int idx = 0; idx < numTads; idx++ ) {
        T* buffer = _buffer + tad->tadOffsets[idx];
        NDArray<T>* array = new NDArray<T>(buffer, shapeInfo);
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
template<typename T>
ResultSet<T>* NDArray<T>::allTensorsAlongDimension(const std::initializer_list<int>& dimensions) const {
        
    return allTensorsAlongDimension(std::vector<int>(dimensions));
}

////////////////////////////////////////////////////////////////////////
template<typename T>
ResultSet<T>* NDArray<T>::allExamples() const {
        
    std::vector<int> dimensions(rankOf() - 1);            
    for (int e = 1; e < rankOf(); e++)
        dimensions[e-1] = e;

    return allTensorsAlongDimension(dimensions);
}


/*

template NDArray<float>* NDArray<float>::asT<float>();
template NDArray<float16>* NDArray<float>::asT<float16>();
template NDArray<double>* NDArray<float>::asT<double>();

template NDArray<float>* NDArray<float16>::asT<float>();
template NDArray<float16>* NDArray<float16>::asT<float16>();
template NDArray<double>* NDArray<float16>::asT<double>();

template NDArray<float>* NDArray<double>::asT<float>();
template NDArray<float16>* NDArray<double>::asT<float16>();
template NDArray<double>* NDArray<double>::asT<double>();


#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

