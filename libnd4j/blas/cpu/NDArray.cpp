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
#include <graph/exceptions/datatype_exception.h>
#include <helpers/ConstantTadHelper.h>

namespace nd4j {


    //////////////////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////////////
    void NDArray::operator delete(void* p) {
        if (!nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            free(p);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    template <>
    utf8string NDArray::e(const Nd4jLong i) const {
        // if (i >= _length)
        //     throw std::invalid_argument("NDArray::e(i): input index is out of array length !");

        if (!isS())
            throw std::runtime_error("This method is available for String arrays only");

        auto rp = getOffset(i);
        return *(reinterpret_cast<utf8string**>(_buffer)[rp]);
    }

    template <>
    std::string NDArray::e(const Nd4jLong i) const {
        auto u = e<utf8string>(i);
        std::string r(u._buffer);
        return r;
    }

    template <typename T>
    T NDArray::e(const Nd4jLong i) const {

        // if (i >= _length)
        //     throw std::invalid_argument("NDArray::e(i): input index is out of array length !");

        auto rp = getOffset(i);

        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), return templatedGet<, T>(this->_buffer, rp), LIBND4J_TYPES);
//        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong) const, LIBND4J_TYPES);


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
        auto result = new NDArray(ordering(), getShapeAsVector(), DataTypeUtils::fromT<T>());
        auto l = this->lengthOf();

        PRAGMA_OMP_PARALLEL_FOR
        for (int e = 0; e < l; e++) {
            result->p(e, this->e<T>(e));
        }

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArray::asT, (), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {
        
    _workspace = other._workspace;
    setShapeInfo(ShapeBuilders::copyShapeInfo(other._shapeInfo, false, _workspace));

    ALLOCATE(_buffer, other._workspace, _length * other.sizeOfT(), int8_t);    

    triggerAllocationFlag(true, true);

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
NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, nd4j::memory::Workspace* workspace, const bool isBuffAlloc, const bool isShapeAlloc) {

    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor:: array can't be initalized without shapeinfo");

    _workspace = workspace;
    _buffer    = reinterpret_cast<int8_t *>(buffer);
    setShapeInfo(shapeInfo);
    triggerAllocationFlag(isBuffAlloc, isShapeAlloc);    
}

////////////////////////////////////////////////////////////////////////
//constructor, create array at given workspace
NDArray::NDArray(nd4j::memory::Workspace* workspace) {
    _buffer    = nullptr;
    _shapeInfo = nullptr;
    _isBuffAlloc = false;
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

    _workspace = workspace;
    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));
    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOf(dtype), int8_t);
    memset(_buffer, 0, _length * DataTypeUtils::sizeOf(dtype));    
    triggerAllocationFlag(true, true);
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _workspace = workspace;
    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));

    if (_length != data.size()) {
        nd4j_printf("NDArray constructor: data size [%i] doesn't match shape length [%i]\n", data.size(), _length);
        throw std::runtime_error("NDArray constructor: data size doesn't match shape");
    }

    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOf(dtype), int8_t);
    
    triggerAllocationFlag(true, true);

    for(Nd4jLong i=0; i < _length; ++i) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedDoubleAssign<, double>(_buffer, i, reinterpret_cast<const void *>(data.data()), i), LIBND4J_TYPES);
    }
}


////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {

    ALLOCATE(_buffer, workspace, other->_length * DataTypeUtils::sizeOf(other->dataType()), int8_t);
    _workspace = workspace;
    setShapeInfo(ShapeBuilders::copyShapeInfo(other->_shapeInfo, copyStrides, workspace));    
    triggerAllocationFlag(true, true);

    // memcpy is handled within execTransformAny
    NativeOpExcutioner::execTransformAny(transform::AnyOps::Assign, other->_buffer, other->_shapeInfo, _buffer, _shapeInfo, nullptr, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape,  nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _workspace = workspace;
    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));

    _buffer = reinterpret_cast<int8_t *>(buffer);    
    triggerAllocationFlag(false, true);
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(Nd4jLong* shapeInfo, const bool copyStrides, nd4j::memory::Workspace* workspace, const bool isShapeAlloc) {
   
    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _workspace = workspace;

    if(isShapeAlloc) {
        setShapeInfo(shapeInfo);
        _isShapeAlloc = true;
        if(!copyStrides)
            shape::updateStrides(_shapeInfo, shape::order(shapeInfo));                
    }
    else {
        setShapeInfo(ShapeBuilders::copyShapeInfo(shapeInfo, copyStrides, workspace));        
        _isShapeAlloc = true;
        if(isShapeAlloc)
            RELEASE(shapeInfo, workspace);        
    }    
    
    if (!isEmpty()) {        
        ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOfElement(_dataType), int8_t);
        memset(_buffer, 0, _length * DataTypeUtils::sizeOfElement(_dataType));
        _isBuffAlloc = true;
    }    
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros, set dtype as array type
NDArray::NDArray(Nd4jLong* shapeInfo, const nd4j::DataType dtype, const bool copyStrides, nd4j::memory::Workspace* workspace, const bool isShapeAlloc) {

    if (shapeInfo == nullptr || (int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("NDArray constructor: input shapeInfo is nullptr or its rank exceeds 32");


    _workspace = workspace;

    if(isShapeAlloc) {
        setShapeInfo(shapeInfo, dtype);
        _isShapeAlloc = true;
        if(!copyStrides)
            shape::updateStrides(_shapeInfo, shape::order(shapeInfo));                
    }
    else {
        setShapeInfo(ShapeBuilders::copyShapeInfoAndType(shapeInfo, dtype, copyStrides, workspace));        
        _isShapeAlloc = true;
        if(isShapeAlloc)
            RELEASE(shapeInfo, workspace);        
    }      

    if(!isEmpty()) {
        ALLOCATE(_buffer, _workspace, _length * sizeOfT() , int8_t);    
        memset(_buffer, 0, _length * DataTypeUtils::sizeOfElement(_dataType));
        _isBuffAlloc = true;
    }
}

////////////////////////////////////////////////////////////////////////
// creates scalar or empty array depending on bool argument isScalar
NDArray::NDArray(nd4j::DataType dtype, nd4j::memory::Workspace* workspace, const bool isScalar) {

    _workspace = workspace;

    if(isScalar) {
        setShapeInfo(ShapeBuilders::createScalarShapeInfo(dtype, workspace));
        ALLOCATE(_buffer, workspace, DataTypeUtils::sizeOfElement(dtype), int8_t);
        memset(_buffer, 0, DataTypeUtils::sizeOfElement(dtype));    
        triggerAllocationFlag(true, true);
    }
    else {
        setShapeInfo(ShapeBuilders::emptyShapeInfo(dtype, workspace));            
        triggerAllocationFlag(false, true);        
    }
}


    template <typename T>
    void NDArray::templatedAssign(void *xBuffer, Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        auto x = reinterpret_cast<T *>(xBuffer);
        const auto y = reinterpret_cast<const T*>(yBuffer);
        if (xBuffer != nullptr && yBuffer != nullptr)
        x[xOffset] = y[yOffset];
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES);

    bool NDArray::isC() const {
        // TODO: this method must be implemented once we add support for complex numbers
        return false;
    }

    bool NDArray::isS() const {
        return _dataType == DataType::UTF8;
    }

    bool NDArray::isR() const {
        auto xType = ArrayOptions::dataType(this->_shapeInfo);
        return xType == FLOAT32 || xType == HALF || xType == DOUBLE || xType == FLOAT8;
    }

    bool NDArray::isZ() const {
        // TODO: decide if we really want to exclude Bool here
        return !isC() && !isR() && !isB() && !isS();
    }

    bool NDArray::isB() const {
        return ArrayOptions::dataType(this->_shapeInfo) == BOOL;
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

    template<>
    std::string NDArray::toStringValue(bfloat16 value) {
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
            if (this->isR())
                os << toStringValue(this->e<float>(e));
            else if (this->isZ())
                os << toStringValue(this->e<Nd4jLong>(e));
            else if (this->isB())
                os << toStringValue(this->e<bool>(e));
            else if (this->isS())
                os << this->e<std::string>(e);

            if (e < limit - 1)
                os << ", ";
        }

        os << "]";

        return os.str();
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<T> NDArray::getBufferAsVector() {

        std::vector<T> vector(_length);

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < _length; e++)
            vector[e] = this->e<T>(e);

        return vector;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::getBufferAsVector(), LIBND4J_TYPES);



////////////////////////////////////////////////////////////////////////
    std::vector<Nd4jLong> NDArray::getShapeAsVector() const {
        std::vector<Nd4jLong> vector(this->rankOf());

        for (int e = 0; e < this->rankOf(); e++)
            vector[e] = this->sizeAt(e);

        return vector;
    }

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeInfoAsFlatVector() {
    int magicNumber = shape::shapeInfoLength(this->rankOf());
    std::vector<int64_t> vector;

    for (int e = 0; e < magicNumber; e++)
        vector.emplace_back(static_cast<int64_t>(_shapeInfo[e]));

    return vector;
}

std::vector<int64_t> NDArray::getShapeAsFlatVector() {
    std::vector<int64_t> vector;

    for (int e = 0; e < this->rankOf(); e++)
        vector.emplace_back(static_cast<int64_t>(this->sizeAt(e)));

    return vector;
}

////////////////////////////////////////////////////////////////////////
    std::vector<Nd4jLong> NDArray::getShapeInfoAsVector() {
        int magicNumber = shape::shapeInfoLength(this->rankOf());
        std::vector<Nd4jLong> vector;

        for (int e = 0; e < magicNumber; e++)
            vector.emplace_back(this->_shapeInfo[e]);

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
        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != second->_dataType || _dataType != third->_dataType || _dataType != target->_dataType)
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

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != other->_dataType || _dataType != target->_dataType)
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
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int8_t (int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bool (bool, bool)>& func, NDArray* target);


////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::applyLambda(const std::function<T(T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
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
    template void NDArray::applyLambda(const std::function<int8_t(int8_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<bool(bool)>& func, NDArray* target);

    template<typename T>
    void NDArray::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
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
        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
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
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int8_t (Nd4jLong, int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bool (Nd4jLong, bool, bool)>& func, NDArray* target);
#endif



////////////////////////////////////////////////////////////////////////
    std::vector<int8_t> NDArray::asByteVector() {
        // string tensors require special treatment
        if (this->dataType() == UTF8) {
            // length + number of elements + last offset
            auto prefixLength = 1 + this->lengthOf() + 1;
            std::vector<Nd4jLong> prefix(prefixLength);

            // total number of string elements goes first
            prefix[0] = this->lengthOf();

            // now rolling through elements, to fill cumulative
            auto dataLength = 0;
            for (Nd4jLong e = 0; e < lengthOf(); e++) {
                auto s = this->e<utf8string>(e);
                prefix[e+1] = dataLength;
                dataLength += s._length;
            }

            // final prefix
            prefix[lengthOf()+1] = dataLength;

            // preallocating all at once
            std::vector<int8_t> result((prefixLength * sizeof(Nd4jLong)) + dataLength);
            auto charPtr = result.data() + (prefixLength * sizeof(Nd4jLong));
            for (int e = 0; e < this->lengthOf(); e++) {
                auto s = this->e<utf8string>(e);
                auto cPtr = charPtr + prefix[e+1];
                memcpy(cPtr, s._buffer, s._length);
            }

            // copying prefix data to result buffer
            memcpy(result.data(), prefix.data(), prefix.size() * sizeof(Nd4jLong));

            return result;
        } else {
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
    }

    void NDArray::linspace(const double start) {
        linspace(start, 1);
    }

    void NDArray::linspace(const double start, const double step) {
        if (isS())
            throw std::runtime_error("NDArray::linspace: you can't use this method on String array!");

        Nd4jLong numElements = this->lengthOf();

        for (Nd4jLong e = 0; e < numElements; e++) {
            this->p(e, start + (step * e));
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

////////////////////////////////////////////////////////////////////////
// assignment operator
    NDArray& NDArray::operator=(const NDArray& other) {

	if (this == &other)
        return *this;

    if (_shapeInfo != nullptr && _buffer != nullptr && shape::equalsSoft(_shapeInfo, other._shapeInfo) && _dataType == other._dataType) {
        this->assign(&other);
    }
    else {
        if(_isBuffAlloc && _workspace == nullptr)
            delete []_buffer;
        
        _workspace = other._workspace;
        setShapeInfo(ShapeBuilders::copyShapeInfo(other._shapeInfo, false, _workspace));
        
        ALLOCATE(_buffer, _workspace, _length * sizeOfT(), int8_t);

        triggerAllocationFlag(true, true);
        
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
    _length       = other._length;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;

    return *this;
}

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray& NDArray::operator=(const T scalar) {
        if (this->_buffer == nullptr) {
            ALLOCATE(_buffer, _workspace, DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>()), int8_t);
            _isBuffAlloc = true;
        }

        if (this->_shapeInfo == nullptr) {
            auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), _workspace);
            this->setShapeInfo(shapeInfo);
            _isShapeAlloc = true;
        }

        this->assign(scalar);
        return *this;
    }
    template ND4J_EXPORT NDArray& NDArray::operator=<double>(const double scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<float>(const float scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<float16>(const float16 scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<bfloat16>(const bfloat16 scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<Nd4jLong>(const Nd4jLong scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<int>(const int scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<int8_t>(const int8_t scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<uint8_t>(const uint8_t scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<int16_t>(const int16_t scalar);
    template ND4J_EXPORT NDArray& NDArray::operator=<bool>(const bool scalar);


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
    void NDArray::templatedDoubleAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        auto x = reinterpret_cast<X *>(xBuffer);
        const auto y = reinterpret_cast<const Y *>(yBuffer);

        x[xOffset] = static_cast<X>(y[yOffset]);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedDoubleAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES, LIBND4J_TYPES);

// This method assigns values of given NDArray to this one, wrt order
    void NDArray::assign(const NDArray *other) {
        assign(*other);
    }

// This method assigns values of given NDArray to this one
    void NDArray::assign(const NDArray& other) {

        if (this == &other)
            return;

        if (other.isScalar()) {
            if(this->isScalar()) {
                if (!this->isEmpty() && !other.isEmpty()) {
                    BUILD_DOUBLE_SELECTOR(_dataType, other._dataType, templatedDoubleAssign, (_buffer, 0, other._buffer, 0), LIBND4J_TYPES, LIBND4J_TYPES);
                }
                else if (this->isEmpty() != other.isEmpty()) { // need assign non-empty scalar to empty
                    if (other.isEmpty())
                        ArrayOptions::setPropertyBit(this->_shapeInfo, ARRAY_EMPTY);
                    else
                        *this = other;
                }
            }
            else {
                if (Environment::getInstance()->isExperimentalBuild() || this->dataType() == other.dataType() || other.isB()) {
                    NativeOpExcutioner::execScalar(scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, other._buffer, other._shapeInfo, nullptr);
                } else {
                    if (other.isR()) {
                        auto o = other.e<double>(0);
                        this->assign(o);
                    } else if (other.isZ()) {
                        auto o = other.e<Nd4jLong>(0);
                        this->assign(o);
                    } else
                        throw datatype_exception::build("NDArray::assign() requires data types must be the same", this->dataType(), other.dataType());
                }
            }
            return;
        }

        if (other._length != _length) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && _dataType == other._dataType && ews() == 1 && other.ews() == 1)
            memcpy(_buffer, other._buffer, _length * sizeOfT());
        else
            NativeOpExcutioner::execTransformAny(transform::Assign, other._buffer, other._shapeInfo, _buffer, _shapeInfo, nullptr, nullptr, nullptr);

    }


// This method assigns given value to all elements in this NDArray
    void NDArray::assign(const double value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->_dataType, value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const float value) {
        // just fire scalar
        NDArray temp(this->dataType(), this->_workspace);
        //*reinterpret_cast<float*>(temp.buffer()) = value;
        temp.p(0, value);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const float16 value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const bfloat16& value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const Nd4jLong value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int16_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const uint8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const int8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    void NDArray::assign(const bool value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_workspace);
        NativeOpExcutioner::execScalar(nd4j::scalar::CopyPws, _buffer, _shapeInfo, _buffer, _shapeInfo, temp.buffer(), temp.shapeInfo(), nullptr);
    }

    NDArray* NDArray::detach() {

        if (!isAttached())
            return this;

        void* newBuffer;

        newBuffer = new int8_t[lengthOf() * sizeOfT()];

        Nd4jLong* newShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, false, nullptr);

        auto result = new NDArray(newBuffer, newShapeInfo, nullptr, true, true);

        result->assign(this);

        return result;

    }

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
    NDArray* NDArray::dup(const char newOrder) {

    char order = newOrder == 'a' ? ordering() : newOrder;

    if (isEmpty()) {
        return NDArrayFactory::empty_(this->dataType(), this->_workspace);
    }

    // for now string arrays require special treatment
    if (this->dataType() == DataType::UTF8) {
        std::vector<std::string> strings(_length);
        for (int e = 0; e < _length; e++)
            strings[e] = this->e<std::string>(e);

        auto result = NDArrayFactory::string_(order, this->getShapeAsVector(), strings, _workspace);
        return result;
    } else {

        auto outShapeInfo = ShapeBuilders::createShapeInfo(_dataType, order, getShapeAsVector(), _workspace);
        void *outBuffer = nullptr;
        ALLOCATE(outBuffer, _workspace, _length * sizeOfT(), int8_t);

        auto result = new NDArray(outBuffer, outShapeInfo, _workspace, true, true);
        result->assign(this);

        return result;
    }
}

    NDArray NDArray::varianceNumber(nd4j::variance::Ops op, bool biasCorrected) {
        NDArray res(DataTypeUtils::pickFloatingType(dataType()), _workspace);
        NativeOpExcutioner::execSummaryStatsScalar(op,this->getBuffer(), this->getShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), biasCorrected);
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    NDArray NDArray::sumNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::sumNumber: you can't use this method on String array!");
        NDArray res(_dataType, _workspace);
        NativeOpExcutioner::execReduceSameScalar(nd4j::reduce::SameOps::Sum, _buffer, _shapeInfo, nullptr, res.buffer(), res.shapeInfo());
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    NDArray NDArray::meanNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::meanNumber: you can't use this method on String array!");
        NDArray res(DataTypeUtils::pickFloatingType(dataType()), _workspace);
        NativeOpExcutioner::execReduceFloatScalar(nd4j::reduce::FloatOps::Mean, _buffer, _shapeInfo, nullptr, res.buffer(), res.shapeInfo());
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
        if (isS())
            throw std::runtime_error("NDArray::hasNaNs: you can't use this method on String array!");
        return this->reduceNumber(nd4j::reduce::IsNan, nullptr).e<int>(0) > 0;
    }

    bool NDArray::hasInfs() {
        if (isS())
            throw std::runtime_error("NDArray::hasInfs: you can't use this method on String array!");
        return this->reduceNumber(nd4j::reduce::IsInf, nullptr).e<int>(0) > 0;
    }

    bool NDArray::isFinite() {
        if (isS())
            throw std::runtime_error("NDArray::isFinite: you can't use this method on String array!");
        return this->reduceNumber(nd4j::reduce::IsInfOrNan, nullptr).e<int>(0) == 0;
    }

    template <typename T, typename Y>
    void NDArray::templatedSet(void *buffer, const Nd4jLong *indices, const void *value) {
        auto t = reinterpret_cast<T *>(buffer);
        const auto y = *(reinterpret_cast<const Y *>(value));

        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), indices, rankOf());
        t[xOffset] = static_cast<T>(y);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, const void *value), LIBND4J_TYPES, LIBND4J_TYPES);

    template <typename T, typename Y>
    void NDArray::templatedSet(void *buffer, const Nd4jLong offset, const void *value) {
        auto t = reinterpret_cast<T *>(buffer);
        const auto y = *(reinterpret_cast<const Y *>(value));

        t[offset] = static_cast<T>(y);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong offset, const void *value), LIBND4J_TYPES, LIBND4J_TYPES);


    void NDArray::setWorkspace(memory::Workspace* workspace) {
        this->_workspace = workspace;
    }

    void* NDArray::bufferWithOffset(Nd4jLong offset) const {
        
        return _buffer + (offset * sizeOfT());
    }

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
NDArray* NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    return new NDArray(reduceAlongDims(op, dimensions, keepDims, supportOldShapes));
}

NDArray* NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    return new NDArray(reduceAlongDims(op, dimensions, keepDims, supportOldShapes));
}

NDArray* NDArray::reduceAlongDimension(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    return new NDArray(reduceAlongDims(op, dimensions, keepDims, supportOldShapes));
}

NDArray* NDArray::reduceAlongDimension(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    return new NDArray(reduceAlongDims(op, dimensions, keepDims, supportOldShapes));
}

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
NDArray NDArray::reduceAlongDims(nd4j::reduce::FloatOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDims FloatOps: you can't use this method on String array!");

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
    if(!isR())
        ArrayOptions::setDataType(newShape, Environment::getInstance()->defaultFloatDataType());

    NDArray result(newShape, true, _workspace);
    RELEASE(newShape, _workspace);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDims SameOps: you can't use this method on String array!");

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);

    NDArray result(newShape, true, _workspace);
    RELEASE(newShape, _workspace);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDims BoolOps: you can't use this method on String array!");

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
    if(!isB())
        ArrayOptions::setDataType(newShape, DataType::BOOL);

    NDArray result(newShape, true, _workspace);
    RELEASE(newShape, _workspace);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDims LongOps: you can't use this method on String array!");

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
    if(_dataType != DataType::INT64)
        ArrayOptions::setDataType(newShape, DataType::INT64);

    NDArray result(newShape, true, _workspace);
    RELEASE(newShape, _workspace);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: you can't use this method on String array!");
    if (target == nullptr || !target->isR())
        throw std::invalid_argument("NDArray::reduceAlongDimension FloatOps: requires target array to be present and have type form real space!");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: wrong target shape!");
        RELEASE(newShape, _workspace);
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExcutioner::execReduceFloatScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->buffer(), target->shapeInfo());
    else {
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        NativeOpExcutioner::execReduceFloat(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: you can't use this method on String array!");
    else if (target == nullptr)
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: requires target array to be present");
    else if (target->_dataType != _dataType)
        throw datatype_exception::build("NDArray::reduceAlongDimension SameOps: requires target array to have same dtype as input", _dataType, target->dataType());

        std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension SameOps: wrong target shape!");
        RELEASE(newShape, _workspace);
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExcutioner::execReduceSameScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->buffer(), target->shapeInfo());
    else {
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        NativeOpExcutioner::execReduceSame(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceAlongDimension(nd4j::reduce::BoolOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension BoolOps: you can't use this method on String array!");
    if (target == nullptr || !target->isB())
        throw std::invalid_argument("NDArray::reduceAlongDimension BoolOps: requires target array to be present and have BOOL type!");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension BoolOps: wrong target shape!");
        RELEASE(newShape, _workspace);
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExcutioner::execReduceBoolScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->buffer(), target->shapeInfo());
    else {
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        NativeOpExcutioner::execReduceBool(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceAlongDimension(nd4j::reduce::LongOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != DataType::INT64)
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps: requires target array to be present and have type of INT64");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension LongOps: wrong target shape!");
        RELEASE(newShape, _workspace);
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExcutioner::execReduceLongScalar(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->buffer(), target->shapeInfo());
    else {
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        NativeOpExcutioner::execReduceLong(op, this->getBuffer(), this->getShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
NDArray *NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

NDArray *NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

NDArray *NDArray::reduceAlongDimension(nd4j::reduce::BoolOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

NDArray *NDArray::reduceAlongDimension(nd4j::reduce::LongOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}


//////////////////////////////////////////////////////////////////////////
    NDArray NDArray::reduceNumber(nd4j::reduce::FloatOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), this->_workspace);
        NDArray result(shape, true, this->_workspace);
        RELEASE(shape, this->_workspace);

        NativeOpExcutioner::execReduceFloatScalar(op, _buffer, _shapeInfo, extraParams, result.buffer(), result.shapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");

        NDArray result(_dataType, _workspace);
        NativeOpExcutioner::execReduceSameScalar(op, _buffer, _shapeInfo, extraParams, result.buffer(), result.shapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::BOOL, this->_workspace);
        NDArray result(shape, true, this->_workspace);
        RELEASE(shape, this->_workspace);

        NativeOpExcutioner::execReduceBoolScalar(op, _buffer, _shapeInfo, extraParams, result.buffer(), result.shapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::LongOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::INT64, this->_workspace);
        NDArray result(shape, true, this->_workspace);
        RELEASE(shape, this->_workspace);

        NativeOpExcutioner::execReduceLongScalar(op, _buffer, _shapeInfo, extraParams, result.buffer(), result.shapeInfo());
        return result;
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::reduceNumber(nd4j::reduce::FloatOps op, NDArray& target, void *extraParams) const {
        
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataTypeUtils::pickFloatingType(_dataType))
            throw std::invalid_argument("NDArray::reduceNumber FloatOps: target array should be scalar and have corresponding float type!");

        NativeOpExcutioner::execReduceFloatScalar(op, _buffer, _shapeInfo, extraParams, target._buffer, target._shapeInfo);        
    }

    void NDArray::reduceNumber(nd4j::reduce::SameOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != _dataType)
            throw std::invalid_argument("NDArray::reduceNumber SameOps: target array should be scalar and have same type as this array!");
        
        NativeOpExcutioner::execReduceSameScalar(op, _buffer, _shapeInfo, extraParams, target._buffer, target._shapeInfo);
    }

    void NDArray::reduceNumber(nd4j::reduce::BoolOps op, NDArray& target, void *extraParams) const {
        
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::reduceNumber BoolOps: target array should be scalar and have bool type!");
        
        NativeOpExcutioner::execReduceBoolScalar(op, _buffer, _shapeInfo, extraParams, target._buffer, target._shapeInfo);
    }

    void NDArray::reduceNumber(nd4j::reduce::LongOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::INT64)
            throw std::invalid_argument("NDArray::reduceNumber LongOps: target array should be scalar and have long type!");        

        NativeOpExcutioner::execReduceLongScalar(op, _buffer, _shapeInfo, extraParams, target._buffer, target._shapeInfo);
    }

//////////////////////////////////////////////////////////////////////////
    NDArray NDArray::indexReduceNumber(nd4j::indexreduce::Ops op, void *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::indexReduceNumber: you can't use this method on String array!");

        auto res = NDArrayFactory::create<Nd4jLong>(0);
        NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, extraParams, res.buffer(), res.shapeInfo());
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    void NDArray::applyTransform(nd4j::transform::FloatOps op, NDArray *target, void *extraParams) {

        if (isS())
            throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isR())
            throw std::runtime_error("NDArray::applyTransform FloatOps: target array must have one of FLOAT types");

        NativeOpExcutioner::execTransformFloat(op, _buffer, _shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

    void NDArray::applyTransform(nd4j::transform::AnyOps op, NDArray *target, void *extraParams) {

        if (isS())
            throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        NativeOpExcutioner::execTransformFloat(op, _buffer, _shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

    void NDArray::applyTransform(nd4j::transform::SameOps op, NDArray *target, void *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform SameOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (target->dataType() != this->dataType())
            throw std::runtime_error("NDArray::applyTransform SameOps: target array must the same data type as original array");

        NativeOpExcutioner::execTransformSame(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

    void NDArray::applyTransform(nd4j::transform::BoolOps op, NDArray *target, void *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform BoolOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isB())
            throw std::runtime_error("NDArray::applyTransform BoolOps: target array must have one of BOOL types");

        NativeOpExcutioner::execTransformBool(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

    void NDArray::applyTransform(nd4j::transform::StrictOps op, NDArray *target, void *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform StrictOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!this->isR() || !target->isR() || (this->dataType() != target->dataType()))
            throw std::runtime_error("NDArray::applyTransform StrictOps: both Source and Target array must have same FLOAT type !");

        NativeOpExcutioner::execTransformStrict(op, this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams, nullptr, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    // void NDArray::applyTransform(nd4j::transform::FloatOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::AnyOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::SameOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::BoolOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::StrictOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // perform array transformation
    NDArray NDArray::transform(nd4j::transform::FloatOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform FloatOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), DataTypeUtils::pickFloatingType(dataType()), this->_workspace);
        NativeOpExcutioner::execTransformFloat(op, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

        NDArray result(this->_shapeInfo, false, this->_workspace);
        NativeOpExcutioner::execTransformSame(op, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::StrictOps op, void *extraParams) const {
        if (!this->isR())
            throw std::runtime_error("Source array must have one of FLOAT types");

        NDArray result(this->_shapeInfo, false, this->_workspace);
        NativeOpExcutioner::execTransformStrict(op, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), nd4j::DataType::BOOL, this->_workspace);
        NativeOpExcutioner::execTransformBool(op, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
    }

//////////////////////////////////////////////////////////////////////////
// perform pairwise transformation
void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray& other, void *extraParams) {
    applyPairwiseTransform(op, &other, this, extraParams);
}

void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray* other, NDArray *target, void *extraParams) const{
    if (isS())
        throw std::runtime_error("NDArray::applyPairwiseTransform: you can't use this method on String array!");
    if (other->lengthOf() != target->lengthOf())
        throw std::invalid_argument("NDArray::applyPairwiseTransform method - lengths of arrays are mismatched");
    if (target->_dataType != this->_dataType && target->_dataType != other->_dataType)
        throw std::invalid_argument("NDArray::applyPairwiseTransform method - type of target array must be the same as type of this or other array !");
    auto otherShape = other->_shapeInfo;
    auto otherBuffer = other->_buffer;
    NDArray* otherAgain = nullptr;
    DataType oldType = ArrayOptions::dataType(otherShape);
    if (!Environment::getInstance()->isExperimentalBuild()) {
        if (dataType() != other->dataType() && other->dataType() != BOOL) {
            Nd4jLong* newShape = nullptr;
            COPY_SHAPE_EX(otherShape, newShape, _workspace);
            ArrayOptions::setDataType(newShape, this->dataType());
            otherAgain = new NDArray(newShape, this->_workspace);
            otherAgain->assign(*other);
            otherBuffer = otherAgain->_buffer;
            otherShape = newShape;
        }
    }
    NativeOpExcutioner::execPairwiseTransform(op, this->_buffer, this->_shapeInfo, otherBuffer, otherShape, target->_buffer, target->_shapeInfo, extraParams);
    if (oldType != ArrayOptions::dataType(otherShape))
        RELEASE(otherShape, _workspace);
        //ArrayOptions::setDataType(otherShape, oldType);

    delete otherAgain;
}

void NDArray::applyPairwiseTransform(nd4j::pairwise::BoolOps op, const NDArray *other, NDArray *target, void *extraParams) const{
    if (isS())
        throw std::runtime_error("NDArray::applyPairwiseTransform BoolOps: you can't use this method on String array!");
    if (other->lengthOf() != target->lengthOf())
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - lengths of arrays are mismatched");
    if (!target->isB())
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - result must have bool type");
    if (_dataType != other->_dataType)
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - this and other arrays must have the same type !");

    auto otherShape = other->_shapeInfo;
    auto otherBuffer = other->_buffer;
    NDArray* otherAgain = nullptr;
    if (!Environment::getInstance()->isExperimentalBuild()) {
        if (dataType() != other->dataType() && other->dataType() != BOOL) {
            ArrayOptions::setDataType(otherShape, this->dataType());
            otherAgain = new NDArray(otherShape, this->_workspace);
            otherAgain->assign(*other);
            otherBuffer = otherAgain->_buffer;
        }
    }
    NativeOpExcutioner::execPairwiseBoolTransform(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, target->_buffer, target->_shapeInfo, extraParams);
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

    void NDArray::printBuffer(const char* msg, Nd4jLong limit) const{
        if (limit == -1)
            limit = (int) this->lengthOf();

        if (msg != nullptr)
            printf("%s: [", msg);
        else
            printf("[");
        if (this->isR()) {
            for (Nd4jLong e = 0; e < limit; e++) {
                printf("%f", this->r<float>(e));
                if (e < limit - 1)
                    printf(", ");
            }
        } else if (this->isZ()) {
            for (Nd4jLong e = 0; e < limit; e++) {
                printf("%llu", this->r<Nd4jLong>(e));
                if (e < limit - 1)
                    printf(", ");
            }
        } else if (this->isB()) {
            for (Nd4jLong e = 0; e < limit; e++) {
                if (this->r<bool>(e))
                    printf("true");
                else
                    printf("false");

                if (e < limit - 1)
                    printf(", ");
            }
        }
        else if (this->isS()) {
            for (Nd4jLong e = 0; e < limit; e++) {
                printf("\"%s\"", this->e<utf8string>(e)._buffer);
                if (e < limit - 1)
                    printf(", ");
            }
        }
        printf("]\n");
        fflush(stdout);
    }

    static void printFormatted(NDArray const* arr, int depth, int limit) {
        if (arr->rankOf() == 2) {
            Nd4jLong rows = arr->rows();
            Nd4jLong cols = arr->columns();
            char* padding = new char[depth + 1];
            memset(padding, ' ', depth);
            padding[depth] = 0;
            printf("[");
            for (Nd4jLong row = 0; row < rows; ++row) {
                if (row && depth > 0)
                    printf("%s", padding);
                printf("[");
                Nd4jLong colLimit = cols > limit?cols:limit;
                for (Nd4jLong col = 0; col < colLimit; ++col) {
                    if (col)
                        printf(" ");
                    if (arr->isR())
                        printf("%f", arr->e<float>(row, col));
                    else if (arr->isZ())
                        printf("%lld", arr->e<Nd4jLong>(row, col));
                    else if (arr->isB())
                        printf("%s", arr->e<bool>(row, col)?"true":"false");
                    else if (arr->isS()) {
                        printf("\"%s\"", arr->e<std::string>(row * cols + col).c_str());
                    }
                }
                if (row < rows - 1)
                    printf("]\n");
                else
                    printf("]");
            }
            printf("]");

            if (padding)
                delete [] padding;
        }
        else {
            //std::unique_ptr<ResultSet> arrs(arr->allTensorsAlongDimension({0}));
            size_t restCount = 2;

            printf("[");

            restCount = ShapeUtils::getNumOfSubArrs(arr->getShapeInfo(), {0});
            for (size_t arrIndex = 0; arrIndex < restCount; ++arrIndex) {
                NDArray subArr = (*arr)(arrIndex, {0});
                printFormatted(&subArr, depth + 1, limit);
                if (arrIndex < restCount - 1) {

                    for (Nd4jLong i = 1; i < arr->rankOf(); ++i)
                        printf("\n");

                    for (Nd4jLong i = 0; i < depth - 2; ++i)
                        printf(" ");

                }

            }

            printf("]");
        }
    }

    void NDArray::printIndexedBuffer(const char* msg, Nd4jLong limit) const {
        Nd4jLong rank = this->rankOf();

        bool rowFlag = (rank < 2) || (rank == 2 && this->sizeAt(0) == 1);

        if (msg)
            printf("%s: ", msg);

        if (this->isEmpty()) {
            printf("Empty\n");
        }
        else if (this->rankOf() == 0) {
            if (this->isZ())
                printf("%lld\n", this->e<Nd4jLong>(0));
            else if (this->isR())
                printf("%f\n", this->e<float>(0));
            else if (this->isB()) {
                printf("%s\n", this->e<bool>(0)?"true":"false");
            }
            else if (this->isS()) {
                printf("\"%s\"\n", this->e<std::string>(0).c_str());
            }

        }
        else if (rowFlag)
            printBuffer(nullptr, limit);
        else {
            if (msg)
                printf("\n");
            printFormatted(this, 1, limit);
            printf("\n");
        }
        //printf("\n");
        fflush(stdout);
    }

    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw std::runtime_error("Can't get index higher than total number of TADs");

        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        Nd4jLong* shapeInfo;
        if (_workspace == nullptr) {
            shapeInfo = new Nd4jLong[shape::shapeInfoLength(tadPack.primaryShapeInfo())];
        } else {
            shapeInfo = reinterpret_cast<Nd4jLong *>(_workspace->allocateBytes(shape::shapeInfoByteLength(tadPack.primaryShapeInfo())));
        }
        std::memcpy(shapeInfo, tadPack.primaryShapeInfo(), shape::shapeInfoByteLength(tadPack.primaryShapeInfo()));

        auto array = new NDArray(bufferWithOffset(tadPack.primaryOffsets()[index]), shapeInfo, _workspace);
        array->_isBuffAlloc = false;
        array->_isShapeAlloc = true;
        array->_isView = true;

        return array;
    }

    template <typename T>
    void* NDArray::templatedPointerShift(const Nd4jLong offset) const {
        return reinterpret_cast<T*>(_buffer) + offset;
    }
    BUILD_SINGLE_TEMPLATE(template void* NDArray::templatedPointerShift, (const Nd4jLong offset) const, LIBND4J_TYPES);

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

    NDArray newArr(_buffer, newShapeInfo, _workspace, false, true);

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

    bool NDArray::equalsTo(const NDArray &other, double eps) const {
        return equalsTo(&other, eps);
    }

    void * NDArray::getBufferAsPointer(nd4j::DataType dtype) {
        int8_t *ptr = nullptr;
        ALLOCATE(ptr, _workspace, _length * DataTypeUtils::sizeOfElement(dtype), int8_t);

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

        if (lengthOf() != other->lengthOf())
            return false;

        // we need to be able to compare [1, len] to [len]
        if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
            // FIXME: do something here?
        } else if (!shape::equalsSoft(_shapeInfo, other->_shapeInfo))
            return false;

        auto extras = NDArrayFactory::create(eps, _workspace);
        auto ptr = extras.getBufferAsPointer(nd4j::DataType::FLOAT32);

        NDArray tmp(nd4j::DataType::FLOAT32, _workspace); // scalar = 0

        // we don't need extraparams for this op
        NativeOpExcutioner::execReduce3Scalar(reduce3::EqualsWithEps, _buffer, _shapeInfo, ptr, other->_buffer, other->_shapeInfo, tmp.buffer(), tmp.shapeInfo());

        RELEASE(reinterpret_cast<int8_t *>(ptr), _workspace);

        if (tmp.e<int>(0) > 0)
            return false;

        return true;
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
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
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
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Subtract, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
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
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
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
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Divide, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(), &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());

    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    void NDArray::addiRowVector(const NDArray *row) {

    if (isS())
        throw std::runtime_error("NDArray::addiRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
        throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension = 1;
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, _buffer, _shapeInfo, &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
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
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, target->getBuffer(), target->getShapeInfo(), &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    void NDArray::addiColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::addiColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension = 0;
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo, &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    void NDArray::muliColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::muliColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension = 0;
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimension);

        NativeOpExcutioner::execBroadcast(nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo, &dimension, 1, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
    }


//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, void *extraParams) const {
    if (isS())
            throw std::runtime_error("NDArray::applyScalarArr BoolOps: you can't use this method on String array!");
    if (target == nullptr || !target->isB())
        throw std::invalid_argument("NDArray::applyScalarArr bool method: target is nullptr or has not bool type!");
    if (_dataType != scalar->_dataType) {
        nd4j_printf("This dtype: [%i]; scalar dtype: [%i]\n", this->_dataType, scalar->_dataType);
        throw std::invalid_argument("NDArray::applyScalarArr bool method: this and scalar arrays must have the same type!");
    }
    NativeOpExcutioner::execScalarBool(op, _buffer, _shapeInfo, target->_buffer, target->_shapeInfo, scalar->_buffer, scalar->_shapeInfo, extraParams);
}

template <typename T>
void NDArray::applyScalar(nd4j::scalar::BoolOps op, const T scalar, NDArray *target, void *extraParams) const {

    auto scalarArr = NDArrayFactory::create<T>(scalar, _workspace);
    applyScalarArr(op, &scalarArr, target, extraParams);
}

template <> void NDArray::applyScalar(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, void *extraParams) const { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
template void NDArray::applyScalar<double>(nd4j::scalar::BoolOps op, const double scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<float>(nd4j::scalar::BoolOps op, const float scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<float16>(nd4j::scalar::BoolOps op, const float16 scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<bfloat16>(nd4j::scalar::BoolOps op, const bfloat16 scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<Nd4jLong>(nd4j::scalar::BoolOps op, const Nd4jLong scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<int>(nd4j::scalar::BoolOps op, const int scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<int16_t>(nd4j::scalar::BoolOps op, const int16_t scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<int8_t>(nd4j::scalar::BoolOps op, const int8_t scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<uint8_t>(nd4j::scalar::BoolOps op, const uint8_t scalar, NDArray *target, void *extraParams) const;
template void NDArray::applyScalar<bool>(nd4j::scalar::BoolOps op, const bool scalar, NDArray *target, void *extraParams) const;

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(nd4j::scalar::Ops op, const NDArray* scalar, NDArray* target, void *extraParams) {
    if (isS())
        throw std::runtime_error("NDArray::applyScalarArr: you can't use this method on String array!");
    if (!scalar->isScalar())
        throw std::invalid_argument("NDArray::applyScalarArr method: operand is not a scalar!");
    if(target == nullptr)
        target = this;
    if(target->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, scalar->_shapeInfo) && !(target->_dataType == this->_dataType || target->_dataType == scalar->_dataType))
        throw std::invalid_argument("NDArray::applyScalarArr method: wrong type of target array!");

    if (this->dataType() == scalar->dataType() || Environment::getInstance()->isExperimentalBuild())
        NativeOpExcutioner::execScalar(op, _buffer, _shapeInfo, target->_buffer, target->_shapeInfo, scalar->_buffer, scalar->_shapeInfo, extraParams);
    else {
        auto tmp = const_cast<NDArray*>(scalar)->cast(this->dataType());
        NativeOpExcutioner::execScalar(op, _buffer, _shapeInfo, target->_buffer, target->_shapeInfo, tmp->_buffer, tmp->_shapeInfo, extraParams);

        delete tmp;
    }
}

template <typename T>
void NDArray::applyScalar(nd4j::scalar::Ops op, const T scalar, NDArray *target, void *extraParams) {

    auto scalarArr = NDArrayFactory::create<T>(this->dataType(), scalar, this->_workspace);
    applyScalarArr(op, &scalarArr, target, extraParams);
}

template <> void NDArray::applyScalar(nd4j::scalar::Ops op, const NDArray* scalar, NDArray *target, void *extraParams) { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
template void NDArray::applyScalar(nd4j::scalar::Ops op, const double scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const float scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const float16 scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const bfloat16 scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const Nd4jLong scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const int scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const int16_t scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const int8_t scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const uint8_t scalar, NDArray *target, void *extraParams);
template void NDArray::applyScalar(nd4j::scalar::Ops op, const bool scalar, NDArray *target, void *extraParams);


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
        int dimSize = dimensions.size();
        for (int e = 0; e < dimSize; e++)
            prod *= dimensions[e];

        if (prod != this->lengthOf()) {
            std::string current = ShapeUtils::shapeAsString(this);
            std::string enforced = ShapeUtils::shapeAsString(dimensions);
            nd4j_printf("Can't enforce new shape, lengths mismatch. Original shape: %s; Requested shape: %s\n", current.c_str(), enforced.c_str());
            throw std::runtime_error("Incompatible shape");
        }
        
        char order = o == 'a' ? this->ordering() : o;

        Nd4jLong *newShape = ShapeBuilders::createShapeInfo(dataType(), order, dimensions, _workspace);        

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);

        _shapeInfo = newShape;
        _isShapeAlloc = true;
    }

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
    bool NDArray::reshapei(const char order, const std::vector<Nd4jLong>& cshape) {    

    // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary 
    if(order == ordering() && shape::shapeEquals(rankOf(), shapeOf(), cshape.size(), cshape.data()))         
        return true;

    std::vector<Nd4jLong> shape(cshape);
    int rank = shape.size();

    // looking for negative in shape

    int numberNegativesOnes = 0;

    Nd4jLong* shape_ = shape.data();
    int shapeSize = shape.size();
    for (int i = 0; i < shapeSize; i++) {
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

            for (int j = 0; j < shapeSize; j++)
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

    Nd4jLong arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(_buffer==nullptr || arrLength != this->lengthOf()) {
        this->printShapeInfo("Mismatched shape");
        nd4j::Logger::printv("Shape requested: ", shape);
        nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
        throw std::runtime_error("Bad shape!");
    }
    
    Nd4jLong *shapeInfoNew;
    ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), Nd4jLong);

    bool canReshape = shape::reshapeC(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), shapeInfoNew);
    
    // we can do this only if there was no permute applied, or there are no weird strides
    if (canReshape) {
        
        if(ordering() == 'c' && order == 'f')
            throw std::invalid_argument("NDArray::reshapei(order, shape): in case of reshapeC it doesn't make sense to reshape from c order to f order !");

        shape::setEws(shapeInfoNew, arrLength);
        setShapeInfo(shapeInfoNew);        
        _isShapeAlloc = true;
    } 
    else {

        NDArray temp(order, shape, dataType(), _workspace);
        this->applyTransform(transform::Copy, &temp, nullptr);
        *this = std::move(temp);
        RELEASE(shapeInfoNew, _workspace);
    }

    return canReshape;
}

//////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::argMax(std::initializer_list<int> dimensions) {
        if (isS())
            throw std::runtime_error("NDArray::argMax: you can't use this method on String array!");

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
        Nd4jLong* newShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, true, _workspace);
        auto newArr = new NDArray(_buffer, newShapeInfo, _workspace, false, true);

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
        NDArray result(newBuff, newShapeInfo, _workspace, true, true);

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto resultLen = result.lengthOf();
        auto xType = this->dataType();
        if(result.ordering() == 'c') {           //  ews == 1 always here

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(Nd4jLong i=0;  i<resultLen; ++i) {
                auto yOffset = shape::subArrayOffset(i, newShapeInfo, _shapeInfo);
                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (newBuff, i, this->_buffer, yOffset), LIBND4J_TYPES);

            }
        }
        else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(int i=0;  i<resultLen; ++i) {
                auto xOffset = result.getOffset(i);
                auto yOffset = shape::subArrayOffset(i, newShapeInfo, _shapeInfo);
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
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(Nd4jLong i=0;  i < targetLen; ++i) {
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i*ews, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(int i=0;  i<targetLen; ++i) {

                auto xOffset = target.getOffset(i);
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, xOffset, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
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

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int i = 0; i < targetLen; ++i) {
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i*ews, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for(int i=0;  i<targetLen; ++i) {

                auto xOffset = target.getOffset(i);
                auto yOffset = shape::subArrayOffset(i, target._shapeInfo, _shapeInfo);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, xOffset, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
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

        auto ret = new NDArray('c', outShape, _dataType,  _workspace);

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
    
        if(dimension < 0)
            dimension += rankOf();

        if(rankOf() != target.rankOf())
            throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong rank of target array it must be equal to this array rank!");

        Nd4jLong repeatDelta = target.sizeAt(dimension) / sizeAt(dimension);

        if(repeatDelta == 0)
            throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong shape of target array!");


        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

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
    bool NDArray::permutei(const int* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _workspace);
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, dimensions, _length);
        }

        return true;
    }

    bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

        std::vector<int> copy(dimensions, dimensions + rank);

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(copy.data(), rank, *this, _workspace);
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, copy.data(), _length);
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
        int vecSize = vec.size();

        for (int e = 0; e < vecSize; e++)
            ivec[e] = static_cast<int>(vec[e]);

        return permutei(ivec);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const std::vector<Nd4jLong>& dimensions) {
        std::vector<int> ivec(dimensions.size());

        int dimSize = dimensions.size();
        for (int e = 0; e < dimSize; e++)
            ivec[e] = dimensions[e];

        return permutei(ivec.data(), ivec.size());
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::permute(const int* dimensions, const int rank) const {
        // evaluate shapeInfo for output (permuted) array ret
        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _workspace);
        // create array to be returned
        auto ret = new NDArray(_buffer, shapeInfoNew, _workspace, false, true);
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
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, const std::initializer_list<int> dimensions, const NDArray* other, NDArray* target, void* extraArgs) {
        std::vector<int> vec(dimensions);
        applyBroadcast(op, vec, other, target, extraArgs);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, const std::vector<int>& dimensions, const NDArray* other, NDArray* target, void* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast: you can't use this method on String array!");
        if(((op == broadcast::Divide || op == broadcast::FloorDiv || op == broadcast::FloorMod) && other->isB()) || (op == broadcast::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyBroadcast: you can't divide by array!");
        if(isEmpty() || other->isEmpty()) {
            if(!target->isEmpty())
                throw std::runtime_error("NDArray::applyBroadcast method: when some of input arrays (or both) is empty, target array must be empty as well !");
            return;
        }

        if (dimensions.size() == 0)
            return;
        
        auto result = target == nullptr ? this : target;

        if (other->lengthOf() == lengthOf() && this->rankOf() == other->rankOf()) {
            NativeOpExcutioner::execPairwiseTransform(fromBroadcastToPairwise(op), this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, nullptr);
            return;
        }

        NDArray *min(nullptr), *max(nullptr);
        if((lengthOf() > other->lengthOf()) || (lengthOf() == other->lengthOf() && rankOf() >= other->rankOf()))  {
            max = this;
            min = const_cast<NDArray*>(other);
        }
        else {
            max = const_cast<NDArray*>(other);
            min = this;
        }

        if(result->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, other->_shapeInfo))
            throw std::invalid_argument("NDArray::applyBroadcast method: wrong type of target array !");
        if(!result->isSameShape(max))
            throw std::invalid_argument("NDArray::applyBroadcast method: max and target arrays must have the same shape !");

        std::vector<int> copy(dimensions);

        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(max->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != min->lengthOf())
            throw std::runtime_error("NDArray::applyBroadcast method: tad length mismatch !");

        auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(max->_shapeInfo, copy);
        auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, copy);

        if(max == this)
            NativeOpExcutioner::execBroadcast(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
        else
            NativeOpExcutioner::execInverseBroadcast(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::BoolOps op, const std::vector<int>& dimensions, const NDArray* other, NDArray* target, void* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast BoolOps: you can't use this method on String array!");
        if(isEmpty() || other->isEmpty()) {
            if(!target->isEmpty())
                throw std::runtime_error("NDArray::applyBroadcast method: when some of input arrays (or both) is empty, target array must be empty as well !");
            return;
        }

        if (dimensions.size() == 0)
            return;

        auto result = target == nullptr ? this : target;

        if (other->lengthOf() == lengthOf() && this->rankOf() == other->rankOf()) {
            NativeOpExcutioner::execPairwiseBoolTransform(fromBroadcastToPairwiseBool(op), this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, nullptr);
            return;
        }

        NDArray *min(nullptr), *max(nullptr);
        if((lengthOf() > other->lengthOf()) || (lengthOf() == other->lengthOf() && rankOf() >= other->rankOf()))  {
            max = this;
            min = const_cast<NDArray*>(other);
        }
        else {
            max = const_cast<NDArray*>(other);
            min = this;
        }

        if(result->_dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: type of target array must be BOOL!");
        if(!result->isSameShape(max))
            throw std::invalid_argument("NDArray::applyBroadcast bool method: max and target arrays must have the same shape !");
        if(_dataType != other->_dataType)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: this and other arrays must have the same type !");

        std::vector<int> copy(dimensions);

        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(max->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != min->lengthOf())
            throw std::runtime_error("Tad length mismatch");

        auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(max->_shapeInfo, copy);
        auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, copy);

        // TODO: eventually we want separate tads here
        if(this == max)
            NativeOpExcutioner::execBroadcastBool(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
        else
            NativeOpExcutioner::execInverseBroadcastBool(op, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast bool method: target or other = nullptr !");

        if(isEmpty() || other->isEmpty()) {
            if(!target->isEmpty())
                throw std::runtime_error("NDArray::applyTrueBroadcast method: when some of input arrays (or both) is empty target array must be empty as well !");
            return;
        }
        		
        if (isScalar()) {
            NDArray temp(target->_shapeInfo, _dataType, false, _workspace);
            temp.assign(this);
            temp.applyPairwiseTransform(op.p, other, target,  extraArgs);
            return;
        }
        if (other->isScalar()) {
            this->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        if (other->rankOf() == 1) {
            int dim = 0;
            if (shape::isCommonVector(this->_shapeInfo, dim)) {
                if (this->shapeOf()[dim] == other->lengthOf() && dim == this->rankOf() - 1) {
                    NativeOpExcutioner::execPairwiseBoolTransform(op.p, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, target->_buffer, target->_shapeInfo, extraArgs);
                    return;
                }
            }
        }        

        NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = const_cast<NDArray*>(this);
            min = const_cast<NDArray*>(other);
        }
        else {
            max = const_cast<NDArray*>(other);
            min = const_cast<NDArray*>(this);
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsSoft(target->_shapeInfo, newShapeInfo) || target->_dataType != DataType::BOOL)
                throw std::runtime_error("NDArray::applyTrueBroadcast bool method: the shape or type of target array is wrong !");
            if(_dataType != other->_dataType)
                throw std::invalid_argument("NDArray::applyTrueBroadcast bool method: this and other arrays must have the same type !");

            // if workspace is not null - do not call delete.
            if (_workspace == nullptr)
                delete[] newShapeInfo;
        }

        std::vector<int> maxTadAxes = ShapeUtils::tadAxesForSimpleBroadcast(*max, *min);
        if(!maxTadAxes.empty()) {
            max->applyBroadcast(op.b, maxTadAxes, min, target, extraArgs);
            return;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_workspace);
        
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);

        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = min;
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);

            PRAGMA_OMP_PARALLEL_FOR
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                NDArray targetSubArr = (*target)(i, dimsToExclude);
                if (pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    NDArray pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
        if(pTarget != target)
            delete pTarget;
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast method: target or other = nullptr !");
        if(((op.s == scalar::Divide || op.s == scalar::FloorDiv || op.s == scalar::FloorMod) && other->isB()) || (op.s == scalar::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyTrueBroadcast method: you can't divide by bool array !");

        if (!Environment::getInstance()->isExperimentalBuild()) {
            if (!(this->dataType() == other->dataType() && other->dataType() == target->dataType()))
                throw datatype_exception::build("NDArray::applyTrueBroadcast all ", target->dataType(), this->dataType(), other->dataType());
        }

        if(isEmpty() || other->isEmpty()) {
            if(!target->isEmpty())
                throw std::runtime_error("NDArray::applyTrueBroadcast method: when some of input arrays (or both) is empty target array must be empty as well !");
			return;
		}

        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op.p, *other, extraArgs);
            return;
        }
        if (other->isScalar()) {
            const_cast<NDArray*>(this)->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        if (other->rankOf() == 1) {
            int dim = 0;
            if (shape::isCommonVector(this->_shapeInfo, dim)) {
                if (this->shapeOf()[dim] == other->lengthOf() && dim == this->rankOf() - 1) {
                    NativeOpExcutioner::execPairwiseTransform(op.p, this->_buffer, this->_shapeInfo, other->_buffer, other->_shapeInfo, target->_buffer, target->_shapeInfo, extraArgs);
                    return;
                }
            }
        }


        NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = const_cast<NDArray*>(this);
            min = const_cast<NDArray*>(other);
        }
        else {
            max = const_cast<NDArray*>(other);
            min = const_cast<NDArray*>(this);
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsTypesAndShapesSoft(target->getShapeInfo(), newShapeInfo))
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape or type of target array is wrong !");

            // if workspace is not null - do not call delete.
            if (_workspace == nullptr)
                delete[] newShapeInfo;
        }

        std::vector<int> maxTadAxes = ShapeUtils::tadAxesForSimpleBroadcast(*max, *min);
        if(!maxTadAxes.empty()) {            
            const_cast<NDArray*>(this)->applyBroadcast(op.b, maxTadAxes, other, target, extraArgs);            
            return;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_workspace);        
        
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);

        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = min;
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);

            PRAGMA_OMP_PARALLEL_FOR
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto targetSubArr = (*target)(i, dimsToExclude);
                if(pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    auto pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
         if(pTarget != target)
            delete pTarget;
    }

//////////////////////////////////////////////////////////////////////////
    NDArray NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, void *extraArgs) const {
        Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, _workspace))          // the rank of new array = max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        NDArray result(newShapeInfo, true, this->_workspace);

        // if workspace is not null - do not call delete.
        if (_workspace == nullptr)
            delete[] newShapeInfo;

        this->applyTrueBroadcast(op, &other, &result, false, extraArgs);
  
        return result;
    }


    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, void *extraArgs) const {

        return new NDArray(this->applyTrueBroadcast(op, *other, extraArgs));
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

	    auto ret = new NDArray(shapeInfoNew, true, _workspace);
        ShapeUtils::updateStridesAndType(ret->getShapeInfo(), DataTypeUtils::pickPairwiseResultType(_dataType, other._dataType), order);
	    delete []shapeInfoNew;

    	return ret;
    }


    //////////////////////////////////////////////////////////////////////////
    // check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
    bool NDArray::hasOrthonormalBasis(const int arg) {
        if (isS())
            throw std::runtime_error("NDArray::hasOrthonormalBasis: you can't use this method on String array!");
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

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < this->lengthOf(); e++)
            result[e] = this->e<T>(e);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::asVectorT(), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    // check whether array is identity matrix
    bool NDArray::isIdentityMatrix() {
        if (isS())
            throw std::runtime_error("NDArray::isIdentityMatrix: you can't use this method on String array!");
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
        if (isS())
            throw std::runtime_error("NDArray::isUnitary: you can't use this method on String array!");
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
    T NDArray::r(const Nd4jLong i) const {

        if (i >= _length)
            throw std::invalid_argument("NDArray::r(i): input index is out of array length !");


        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), return templatedGet<, T>(this->_buffer, i), LIBND4J_TYPES);
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::r(const Nd4jLong) const, LIBND4J_TYPES);


//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j) const {
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !");

        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        //return (*this)(i, j);
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {
        //return (*this)(i, j, k);
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray::e(i,j,k): one of input indexes is out of array length or rank!=3 !");

        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

    // returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l) const {
        //return (*this)(i, j, k);
        if (rankOf() != 4)
            throw std::invalid_argument("NDArray::e(i,j,k,l): NDArray isn't rank 4");


        if (i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3]) {
            nd4j_printf("Bad indices are: [%lld, %lld, %lld, %lld]\n", i, j, k, l);
            throw std::invalid_argument("NDArray::e(i,j,k,l): one of input indexes is out of array length!");
        }

        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);

        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::e(const Nd4jLong i) const {
    if (i >= _length)
            throw std::invalid_argument("scalar NDArray::e(i): input index is out of array length !");
    NDArray scalar(_dataType, _workspace);
    BUILD_SINGLE_SELECTOR(_dataType, scalar.templatedSet, (scalar._buffer, 0, dataType(), bufferWithOffset(getOffset(i))), LIBND4J_TYPES);
    return scalar;
}

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
    template <typename T>
    void NDArray::p(const Nd4jLong i, const T value) {

        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, value): input index is out of array length !");

        auto rp = getOffset(i);
        const void *pV = reinterpret_cast<const void*>(const_cast<T *>(&value));
        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), templatedSet<, T>(this->_buffer, rp, pV), LIBND4J_TYPES);
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

    template <>
    std::string* NDArray::bufferAsT() const {
        throw std::runtime_error("This method is NOT supposed to be used");
    }

    template <typename T>
    T * NDArray::bufferAsT() const {
        if (isS())
            throw std::runtime_error("You can't use this method on String array");

        return reinterpret_cast<T*>(_buffer);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template, * NDArray::bufferAsT() const, LIBND4J_TYPES);


    void NDArray::p(const Nd4jLong i, const NDArray& scalar) {
        if(!scalar.isScalar())
            throw std::invalid_argument("NDArray::p method: input array must be scalar!");
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");
        // probably wrong args order
        auto rp = getOffset(i);
        BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (_buffer, rp, scalar.dataType(), scalar.getBuffer()), LIBND4J_TYPES);
        // void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, void *value)
    }


//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {
        //(*this)(i,j) = value;
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
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
   // template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const utf8string value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
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

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::p(i,j,k,l, value): one of input indexes is out of array length or rank!=4 !");
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
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
template <typename T>
void NDArray::pIdx(const Nd4jLong* indices, const T value) {
    for(int i = 0; i < rankOf(); ++i)
        if (indices[i] >= sizeAt(i))
            throw std::invalid_argument("NDArray::pIdx(const Nd4jLong* idx, const T value): input index is out of dimension length !");
    void* v = reinterpret_cast<void *>(const_cast<T *>(&value));

    BUILD_SINGLE_PARTIAL_SELECTOR(_dataType, templatedSet<, T>(_buffer, indices, v), LIBND4J_TYPES);
}

template void NDArray::pIdx(const Nd4jLong* indices, const double value);
template void NDArray::pIdx(const Nd4jLong* indices, const float value);
template void NDArray::pIdx(const Nd4jLong* indices, const float16 value);
template void NDArray::pIdx(const Nd4jLong* indices, const bfloat16 value);
template void NDArray::pIdx(const Nd4jLong* indices, const Nd4jLong value);
template void NDArray::pIdx(const Nd4jLong* indices, const int value);
template void NDArray::pIdx(const Nd4jLong* indices, const int8_t value);
template void NDArray::pIdx(const Nd4jLong* indices, const uint8_t value);
template void NDArray::pIdx(const Nd4jLong* indices, const int16_t value);
template void NDArray::pIdx(const Nd4jLong* indices, const bool value);


    NDArray* NDArray::asT(DataType dtype) {
        if (isS())
            throw std::runtime_error("NDArray::asT: you can't use this method on String array!");
        BUILD_SINGLE_SELECTOR(dtype, return asT, (), LIBND4J_TYPES);
        return nullptr;
    }

    template <typename T>
    NDArray* NDArray::cast() {
        if (isS())
            throw std::runtime_error("NDArray::cast: you can't use this method on String array!");
        return this->asT<T>();
    }
    NDArray* NDArray::cast(DataType dtype) {
        if (isS())
            throw std::runtime_error("NDArray::cast: you can't use this method on String array!");
        return this->asT(dtype);
    }

    void NDArray::cast(NDArray* target, DataType dtype) {
        if (isS())
            throw std::runtime_error("NDArray::cast: you can't use this method on String array!");
        // TODO: to be implemented properly
        target->assign(this);
    }

    void NDArray::applyIndexReduce(nd4j::indexreduce::Ops op, const NDArray* target, const std::vector<int>& dimensions, const void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        if (target->dataType() != nd4j::DataType::INT64)
            throw std::runtime_error("IndexReduce operations return INT64");

        if (target->isScalar()) {
            //target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
            NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, const_cast<void *>(extraParams), target->getBuffer(), target->getShapeInfo());
        } else {
            std::vector<int> copy(dimensions);
            if (dimensions.size() > 1)
                std::sort(copy.begin(), copy.end());

            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExcutioner::execIndexReduce(op, _buffer, _shapeInfo, const_cast<void *>(extraParams), reinterpret_cast<Nd4jLong *>(target->_buffer), target->_shapeInfo, copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
        }
    }
    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    NDArray* NDArray::applyIndexReduce(nd4j::indexreduce::Ops op,const std::vector<int>& dimensions, const void* extraParams ) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        ArrayOptions::setDataType(newShape, nd4j::INT64);
        auto result = new NDArray(newShape, true, _workspace);
        RELEASE(newShape, _workspace);

        if (rankOf() == copy.size()) {
            NativeOpExcutioner::execIndexReduceScalar(op, _buffer, _shapeInfo, const_cast<void *>(extraParams), result->getBuffer(), result->getShapeInfo());
        } else {
            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExcutioner::execIndexReduce(op, _buffer, _shapeInfo, const_cast<void *>(extraParams), reinterpret_cast<Nd4jLong *>(result->_buffer), result->_shapeInfo, copy.data(), copy.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
        }
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const void* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyReduce3 method: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");
        // check shapes consistency
        if(!isSameShape(other))
            throw std::runtime_error("NDArray::applyReduce3 method: the shapes of this and other arrays must be the same !");
        // create shapeInfo for scalar
        Nd4jLong* newShape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), _workspace);
        // create output array (scalar)
        auto result = new NDArray(newShape, true, _workspace, true);
        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = const_cast<void*>(extraParams);
        if(params == nullptr) {
            params = new int8_t[result->sizeOfT()*3];
            memset(params, 0, result->sizeOfT()*3);
        }
        NativeOpExcutioner::execReduce3Scalar(op, _buffer, _shapeInfo, params, other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo);

        if(params != extraParams)
            delete [] static_cast<int8_t*>(params);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const void* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");

        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        ArrayOptions::setDataType(newShape, DataTypeUtils::pickFloatingType(_dataType));
        auto result = new NDArray(newShape, true, _workspace, true);
        // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = const_cast<void*>(extraParams);
        if(params == nullptr) {
            params = new int8_t[result->sizeOfT()*3];
            memset(params, 0, result->sizeOfT()*3);
        }
        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size())
            NativeOpExcutioner::execReduce3Scalar(op, _buffer, _shapeInfo, params, other->_buffer, other->_shapeInfo, result->_buffer, result->shapeInfo());
        else {
            auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);
            auto tadPackY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->_shapeInfo, copy);

            NativeOpExcutioner::execReduce3(op, _buffer, _shapeInfo, params, other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo, copy.data(), copy.size());
        }

        if(params != extraParams)
            delete [] static_cast<int8_t*>(params);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    NDArray* NDArray::applyAllReduce3(nd4j::reduce3::Ops op, const NDArray *other, const std::vector<int>& dimensions, const void* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyAllReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyAllReduce3 method: the types of this and other arrays must be the same !");
        // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               
        // create tads
        auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        auto tadPackY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->_shapeInfo, copy);

        // check tads shapes
        if(!shape::equalsSoft(tadPackX.primaryShapeInfo(), tadPackY.primaryShapeInfo()))
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
        ShapeUtils::updateStridesAndType(newShape, DataTypeUtils::pickFloatingType(_dataType), 'c');
        // create output array
        auto result = new NDArray(newShape, true, _workspace, true);
        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = const_cast<void*>(extraParams);
        if(params == nullptr) {
            params = new int8_t[result->sizeOfT()*3];
            memset(params, 0, result->sizeOfT()*3);

        }

        NativeOpExcutioner::execReduce3All(op, _buffer, _shapeInfo, params, other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo, copy.data(), copy.size(), tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackY.primaryShapeInfo(), tadPackY.primaryOffsets());
        if(params != extraParams)
            delete [] static_cast<int8_t*>(params);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());
            
        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        ArrayOptions::setDataType(newShape, DataTypeUtils::pickFloatingType(_dataType));
        auto result = new NDArray(newShape, true, _workspace, true);
        
        if(rankOf() == copy.size() || copy.empty())
            NativeOpExcutioner::execSummaryStatsScalar(op, _buffer, _shapeInfo, nullptr, result->buffer(), result->shapeInfo(), biasCorrected);
        else
            NativeOpExcutioner::execSummaryStats(op, _buffer, _shapeInfo, nullptr, result->_buffer, result->_shapeInfo, copy.data(), copy.size(), biasCorrected);

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
            return varianceAlongDimension(op, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::varianceAlongDims(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDims: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _workspace);
        ArrayOptions::setDataType(newShape, DataTypeUtils::pickFloatingType(_dataType));
        NDArray result(newShape, true, _workspace, true);

        if(rankOf() == copy.size() || copy.empty())
            NativeOpExcutioner::execSummaryStatsScalar(op, _buffer, _shapeInfo, nullptr, result.buffer(), result.shapeInfo(), biasCorrected);
        else
            NativeOpExcutioner::execSummaryStats(op, _buffer, _shapeInfo, nullptr, result._buffer, result._shapeInfo, copy.data(), copy.size(), biasCorrected);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::varianceAlongDims(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
        return varianceAlongDims(op, biasCorrected, std::vector<int>(dimensions));
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op, const NDArray *target, const bool biasCorrected, const std::vector<int>& dimensions) {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        if (!target->isR())
            throw std::runtime_error("NDArray::varianceAlongDimension: target array must have FLOAT type");

        if(rankOf() == copy.size() || copy.empty())
            NativeOpExcutioner::execSummaryStatsScalar(op, _buffer, _shapeInfo, nullptr, target->getBuffer(), target->getShapeInfo(), biasCorrected);
        else
            NativeOpExcutioner::execSummaryStats(op, _buffer, _shapeInfo, nullptr, target->_buffer, target->_shapeInfo, copy.data(), copy.size(), biasCorrected);
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op,const NDArray *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension(op, target, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    NDArray NDArray::operator()(const std::vector<Nd4jLong>& idx, const bool keepUnitiesInShape, const bool isStrided)  const {

        const int rank = rankOf();
        Nd4jLong *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(rank), Nd4jLong);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(rank));

        auto shapeOf = shape::shapeOf(newShape);
        auto stridesOf = shape::stride(newShape);

        Nd4jLong offset(0), subArrLen(1);        
        int n(isStrided ? 3 : 2), first, last, stride;

        for (int d = rank - 1; d >= 0; --d) {

            if (idx[n * d] != idx[n * d + 1]) {

                first  = idx[n * d]     >= 0 ? idx[n * d]     : idx[n * d]     + sizeAt(d) + 1;
                last   = idx[n * d + 1] >= 0 ? idx[n * d + 1] : idx[n * d + 1] + sizeAt(d) + 1;
                stride = isStrided           ? idx[n * d + 2] : 1;

                shapeOf[d] = (last - first + stride - 1) / stride;      // ceil (last - first) / stride;
                offset += first * stridesOf[d];

                if(shapeOf[d] != 1)
                    stridesOf[d] *= stride;
            }

            subArrLen *= shapeOf[d];
        }
        
        // check if there is possibility to set ews = 1
        shape::setEws(newShape, subArrLen);

        // create resulting sub-array
        NDArray result(bufferWithOffset(offset), newShape, _workspace, false, true);

        if(!keepUnitiesInShape) {
            const int coeff = isStrided ? 3 : 2;
            std::vector<Nd4jLong> nonUnitDims;
            for (int d = 0; d < rank; ++d) {
                if(!(idx[coeff*d] != idx[coeff*d+1] && newShape[d+1] == 1))
                    nonUnitDims.push_back(newShape[d+1]);
            }
            if(nonUnitDims.size() != rank)
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
        if (isS())
            throw std::runtime_error("NDArray::operator+: you can't use this method on String array!");

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), other);
    }

////////////////////////////////////////////////////////////////////////
// addition operator array + scalar
template <typename T>
NDArray NDArray::operator+(const T& scalar) const {
    if (isS())
        throw std::runtime_error("NDArray::operator+: you can't use this method on String array!");
    DataType type;
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = this->dataType();
    }
    auto tmp = NDArrayFactory::create(type, scalar, _workspace);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Add, _buffer, _shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
    return result;
}
template NDArray NDArray::operator+(const double&   scalar) const;
template NDArray NDArray::operator+(const float&    scalar) const;
template NDArray NDArray::operator+(const float16&  scalar) const;
template NDArray NDArray::operator+(const bfloat16&  scalar) const;
template NDArray NDArray::operator+(const Nd4jLong& scalar) const;
template NDArray NDArray::operator+(const int&      scalar) const;
template NDArray NDArray::operator+(const int16_t&  scalar) const;
template NDArray NDArray::operator+(const int8_t&   scalar) const;
template NDArray NDArray::operator+(const uint8_t&  scalar) const;
template NDArray NDArray::operator+(const bool&     scalar) const;

////////////////////////////////////////////////////////////////////////
// subtraction operator array - scalar
template<typename T>
NDArray NDArray::operator-(const T& scalar) const {
    if (isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type;
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = this->dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, _workspace);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Subtract, _buffer, _shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
    return result;
}
template NDArray NDArray::operator-(const double&   scalar) const;
template NDArray NDArray::operator-(const float&    scalar) const;
template NDArray NDArray::operator-(const float16&  scalar) const;
template NDArray NDArray::operator-(const bfloat16&  scalar) const;
template NDArray NDArray::operator-(const Nd4jLong& scalar) const;
template NDArray NDArray::operator-(const int&      scalar) const;
template NDArray NDArray::operator-(const int16_t&  scalar) const;
template NDArray NDArray::operator-(const int8_t&   scalar) const;
template NDArray NDArray::operator-(const uint8_t&  scalar) const;
template NDArray NDArray::operator-(const bool&     scalar) const;

////////////////////////////////////////////////////////////////////////
// multiplication operator array*scalar
template<typename T>
NDArray NDArray::operator*(const T& scalar) const {
    if (isS())
        throw std::runtime_error("NDArray::operator*: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<T>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = this->dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, _workspace);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, _buffer, _shapeInfo, result._buffer, result._shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
    return result;
}
template NDArray NDArray::operator*(const double&   scalar) const;
template NDArray NDArray::operator*(const float&    scalar) const;
template NDArray NDArray::operator*(const float16&  scalar) const;
template NDArray NDArray::operator*(const bfloat16&  scalar) const;
template NDArray NDArray::operator*(const Nd4jLong& scalar) const;
template NDArray NDArray::operator*(const int&      scalar) const;
template NDArray NDArray::operator*(const int16_t&  scalar) const;
template NDArray NDArray::operator*(const int8_t&   scalar) const;
template NDArray NDArray::operator*(const uint8_t&  scalar) const;
template NDArray NDArray::operator*(const bool&     scalar) const;

////////////////////////////////////////////////////////////////////////
// division operator array / scalar
template<typename T>
NDArray NDArray::operator/(const T& scalar) const {
    if (isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");

    if(scalar == (T)0.)
        throw std::runtime_error("NDArray::operator/ (division operator) : division by zero !");

    auto type = DataTypeUtils::fromT<T>();
    if (!Environment::getInstance()->isExperimentalBuild())
        type = this->dataType();

    auto tmp = NDArrayFactory::create(type, scalar, _workspace);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Divide, _buffer, _shapeInfo, result._buffer, result._shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}
template NDArray NDArray::operator/(const double&   scalar) const;
template NDArray NDArray::operator/(const float&    scalar) const;
template NDArray NDArray::operator/(const float16&  scalar) const;
template NDArray NDArray::operator/(const bfloat16&  scalar) const;
template NDArray NDArray::operator/(const Nd4jLong& scalar) const;
template NDArray NDArray::operator/(const int&      scalar) const;
template NDArray NDArray::operator/(const int16_t&  scalar) const;
template NDArray NDArray::operator/(const int8_t&   scalar) const;
template NDArray NDArray::operator/(const uint8_t&  scalar) const;
template NDArray NDArray::operator/(const bool&     scalar) const;

////////////////////////////////////////////////////////////////////////
// addition operator scalar + array
ND4J_EXPORT NDArray operator+(const float16& scalar, const NDArray& arr) {
    return arr + scalar;
}
ND4J_EXPORT NDArray operator+(const bfloat16& scalar, const NDArray& arr) {
     return arr + scalar;
}
ND4J_EXPORT NDArray operator+(const float& scalar, const NDArray& arr) {
    return arr + scalar;
}
ND4J_EXPORT NDArray operator+(const double& scalar, const NDArray& arr) {
    return arr + scalar;
}
ND4J_EXPORT NDArray operator+(const Nd4jLong& scalar, const NDArray& arr) {
    return arr + scalar;
}
ND4J_EXPORT NDArray operator+(const int& scalar, const NDArray& arr) {
    return arr + scalar;
}

////////////////////////////////////////////////////////////////////////
// addition operator scalar + array
ND4J_EXPORT NDArray operator*(const float16& scalar, const NDArray& arr) {
    return arr * scalar;
}
ND4J_EXPORT NDArray operator*(const bfloat16& scalar, const NDArray& arr) {
    return arr * scalar;
}

ND4J_EXPORT NDArray operator*(const float& scalar, const NDArray& arr) {
    return arr * scalar;
}
ND4J_EXPORT NDArray operator*(const double& scalar, const NDArray& arr) {
    return arr * scalar;
}
ND4J_EXPORT NDArray operator*(const Nd4jLong& scalar, const NDArray& arr) {
    return arr * scalar;
}
ND4J_EXPORT NDArray operator*(const int& scalar, const NDArray& arr) {
    return arr * scalar;
}

////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator-(const float16& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<float16>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float16>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator-(const bfloat16& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<bfloat16>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<bfloat16>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const float& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<float>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const double& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");
    DataType type = DataTypeUtils::fromT<double>();
    if (!Environment::getInstance()->isExperimentalBuild()) {
        type = arr.dataType();
    }
    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const Nd4jLong& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<Nd4jLong>();
    if (!Environment::getInstance()->isExperimentalBuild()) {
        type = arr.dataType();
    }
    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<Nd4jLong>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const int& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    DataType type = DataTypeUtils::fromT<int>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}


////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator/(const bfloat16& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    DataType type = DataTypeUtils::fromT<bfloat16>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<bfloat16>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}
////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator/(const float16& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    DataType type = DataTypeUtils::fromT<float16>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float16>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const float& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    DataType type = DataTypeUtils::fromT<float>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const double& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    DataType type = DataTypeUtils::fromT<double>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const int& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    DataType type = DataTypeUtils::fromT<int>();
    if (!Environment::getInstance()->isExperimentalBuild()){
        type = arr.dataType();
    }

    auto tmp = NDArrayFactory::create(type, scalar, arr.getWorkspace());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getWorkspace());
    NativeOpExcutioner::execScalar(nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), result.getBuffer(), result.getShapeInfo(), tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
    return result;
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator+=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExcutioner::execScalar(nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace))
            throw std::invalid_argument("NDArray::operator+=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _workspace);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_workspace);
    }
}

void NDArray::operator-=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator-=: you can't use this method on String array!");

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExcutioner::execScalar(nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace))
            throw std::invalid_argument("NDArray::operator-=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _workspace);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_workspace);
    }
}

void NDArray::operator*=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace))
            throw std::invalid_argument("NDArray::operator*=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _workspace);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_workspace);
    }
}

void NDArray::operator/=(const NDArray& other) {
    if (isS() || other.isS())
        throw std::runtime_error("NDArray::operator/=: you can't use this method on String array!");
    if (other.isB())
        throw std::runtime_error("NDArray::operator/=: you can't divide by bool array!");

    if (this->dataType() != other.dataType() && other.dataType() && !nd4j::Environment::getInstance()->isExperimentalBuild())
        throw nd4j::datatype_exception::build("NDArray operator/= both operands must have same data type", this->dataType(), other.dataType());

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExcutioner::execScalar(nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Divide, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_workspace))
            throw std::invalid_argument("NDArray::operator/=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _workspace);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_workspace);
    }
}
////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::operator+=(const T value) {
    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");

    auto type = DataTypeUtils::fromT<T>();
    if (!Environment::getInstance()->isExperimentalBuild())
        type = this->dataType();


    auto tmp = NDArrayFactory::create(type, value, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
}
template void NDArray::operator+=(const double value);
template void NDArray::operator+=(const float value);
template void NDArray::operator+=(const float16 value);
template void NDArray::operator+=(const bfloat16 value);
template void NDArray::operator+=(const Nd4jLong value);
template void NDArray::operator+=(const int value);
template void NDArray::operator+=(const bool value);

template<typename T>
void NDArray::operator-=(const T value) {
    if (isS())
        throw std::runtime_error("NDArray::operator-=: you can't use this method on String array!");

    auto type = DataTypeUtils::fromT<T>();
    if (!Environment::getInstance()->isExperimentalBuild())
        type = this->dataType();

    auto tmp = NDArrayFactory::create(type, value, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.buffer(), tmp.shapeInfo(), nullptr);
}
template void NDArray::operator-=(const double value);
template void NDArray::operator-=(const float value);
template void NDArray::operator-=(const float16 value);
template void NDArray::operator-=(const bfloat16 value);
template void NDArray::operator-=(const Nd4jLong value);
template void NDArray::operator-=(const int value);
template void NDArray::operator-=(const bool value);

template<typename T>
void NDArray::operator*=(const T scalar) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(this->dataType(), scalar, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
}
template void NDArray::operator*=(const double scalar);
template void NDArray::operator*=(const float scalar);
template void NDArray::operator*=(const float16 scalar);
template void NDArray::operator*=(const bfloat16 scalar);
template void NDArray::operator*=(const Nd4jLong scalar);
template void NDArray::operator*=(const int scalar);
template void NDArray::operator*=(const int16_t scalar);
template void NDArray::operator*=(const int8_t scalar);
template void NDArray::operator*=(const uint8_t scalar);
template void NDArray::operator*=(const bool scalar);

template<typename T>
void NDArray::operator/=(const T scalar) {
    if (isS())
        throw std::runtime_error("NDArray::operator/=: you can't use this method on String array!");

    auto type = DataTypeUtils::fromT<T>();
    if (!Environment::getInstance()->isExperimentalBuild())
        type = this->dataType();

    auto tmp = NDArrayFactory::create(type, scalar, _workspace);
    NativeOpExcutioner::execScalar(nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, tmp.getBuffer(), tmp.getShapeInfo(), nullptr);
}
template void NDArray::operator/=(const double scalar);
template void NDArray::operator/=(const float scalar);
template void NDArray::operator/=(const float16 scalar);
template void NDArray::operator/=(const bfloat16 scalar);
template void NDArray::operator/=(const Nd4jLong scalar);
template void NDArray::operator/=(const int scalar);
template void NDArray::operator/=(const int16_t scalar);
template void NDArray::operator/=(const int8_t scalar);
template void NDArray::operator/=(const uint8_t scalar);
template void NDArray::operator/=(const bool scalar);

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - array
    NDArray NDArray::operator-(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // negative operator, it makes all array elements = -elements
    NDArray NDArray::operator-() const {
        if (isS())
            throw std::runtime_error("NDArray::negative-: you can't use this method on String array!");
        NDArray result(this->_shapeInfo, false, this->_workspace);
        NativeOpExcutioner::execTransformSame(nd4j::transform::Neg, this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, nullptr, nullptr, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*array
    NDArray NDArray::operator*(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator*: you can't use this method on String array!");
        
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }
        if (this->dataType() != other.dataType() && other.dataType() != nd4j::DataType::BOOL && !nd4j::Environment::getInstance()->isExperimentalBuild())
            throw nd4j::datatype_exception::build("NDArray operator* both operands must have same data type", this->dataType(), other.dataType());

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array/array
    NDArray NDArray::operator/(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
        if (other.isB())
            throw std::runtime_error("NDArray::operator/: you can't divide by bool array!");

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_workspace);
            NativeOpExcutioner::execPairwiseTransform(nd4j::pairwise::Divide, _buffer, _shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }
        if (this->dataType() != other.dataType() && other.dataType() != nd4j::DataType::BOOL && !nd4j::Environment::getInstance()->isExperimentalBuild())
            throw nd4j::datatype_exception::build("NDArray operator* both operands must have same data type", this->dataType(), other.dataType());

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // mathematical multiplication of two arrays
    NDArray mmul(const NDArray& left, const NDArray& right) {
        if (left.isS() || right.isS())
            throw std::runtime_error("mmul friend function: you can't use this function on String array!");
        auto ptr = MmulHelper::mmul(const_cast<NDArray*>(&left), const_cast<NDArray*>(&right), nullptr, 1., 0.);
        NDArray result(*ptr);
        delete ptr;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::setIdentity() {
        if (isS())
            throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

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
        PRAGMA_OMP_PARALLEL_FOR_IF(minDim > Environment::getInstance()->tadThreshold())
        for(int i = 0; i < minDim; ++i)
            templatedSet<float>(_buffer, i*offset, this->dataType(), &v);
    }

    template <typename T>
    void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value), LIBND4J_TYPES);



    template <typename T>
    void NDArray::templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
        auto x = reinterpret_cast<T *>(xBuffer);
        auto y = reinterpret_cast<T *>(yBuffer);

        PRAGMA_OMP_PARALLEL_FOR_SIMD
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

        if (isS())
            throw std::runtime_error("NDArray::diagonal: you can't use this method on String array!");
        
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
            outShapeInfo[6] =  0;
        }

        ArrayOptions::setDataType(outShapeInfo, this->dataType());

        auto result = new NDArray(this->_buffer, outShapeInfo, this->_workspace);
        result->_isShapeAlloc = true;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
        if (isS())
            throw std::runtime_error("NDArray::setValueInDiagMatrix: you can't use this method on String array!");
        if(rankOf() != 2)
           throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");

        const auto rows = sizeAt(0);
        const auto cols = sizeAt(1);
        
        switch(direction) {
            
            case 'u':                           // fill upper triangular block
                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
                for(Nd4jLong i = 0; i < rows; ++i)
                    for(Nd4jLong j = 0; j < cols; ++j)
                        if (i + diag <= j)
                            p<T>(i, j, value);
                break;

            case 'l':                           // fill lower triangular block
                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
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
    template void NDArray::setValueInDiagMatrix(const bfloat16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const Nd4jLong& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int16_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const uint8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bool& value, const int diag, const char direction);

    ////////////////////////////////////////////////////////////////////////
    // default destructor
    NDArray::~NDArray() noexcept {
        if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr) {
            if (!isS()) {
                delete[] _buffer;
            } else {
                for (int e = 0; e < lengthOf(); e++) {
                    auto t = reinterpret_cast<utf8string**>(_buffer);
                    delete t[e];
                };

                delete[] _buffer;
            }
        }

        if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr)
            delete[] _shapeInfo;
    }
    
    void NDArray::streamline(char o) {
        char order = o == 'a' ? this->ordering() : o;
    
        int8_t *newBuffer;
        ALLOCATE(newBuffer, this->_workspace, this->lengthOf() * sizeOfT(), int8_t);

        std::vector<Nd4jLong> shape(this->rankOf());
        for (int e = 0; e < this->rankOf(); e++)
            shape[e] = this->sizeAt(e);
        
        Nd4jLong *newShape = ShapeBuilders::createShapeInfo(dataType(), order, shape, _workspace);        

        if (!isView()) {
            NativeOpExcutioner::execTransformSame(transform::Copy, _buffer, _shapeInfo, newBuffer, newShape, nullptr, nullptr, nullptr);
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
            NativeOpExcutioner::execTransformSame(transform::Copy, _buffer, _shapeInfo, newBuffer, newShape, nullptr, nullptr, nullptr);

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

        tileToShape(std::vector<Nd4jLong>(shape), target);
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::tileToShape(const Nd4jLong* shapeInfo) {

        NDArray result(const_cast<Nd4jLong*>(shapeInfo), false, _workspace, false);
        tile(result);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    double NDArray::getTrace() const {
        if (isS())
            throw std::runtime_error("NDArray::getTrace: you can't use this method on String array!");

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

        PRAGMA_OMP_PARALLEL_FOR_REDUCTION(sumT:sum)
        for(int i = 0; i < minDim; ++i)
            sum += e<double>(i*offset);

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

        if(array->isR())
            throw std::invalid_argument("NDArray::quantize: type of array should be from real space!");

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

        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        // FIXME: why we're not using workspaces here?
        Nd4jLong* shapeInfo = new Nd4jLong[shape::shapeInfoLength(tadPack.primaryShapeInfo())];
        std::memcpy(shapeInfo, tadPack.primaryShapeInfo(), shape::shapeInfoByteLength(tadPack.primaryShapeInfo()));

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("NDArray::multipleTensorsAlongDimension: index %i is higher then number of TADs: %i\n", idx, numTads);
                throw std::runtime_error("Bad index");
            }

            auto array = new NDArray(bufferWithOffset(tadPack.primaryOffsets()[idx]), shapeInfo);
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
    ResultSet NDArray::allTensorsAlongDims(const std::vector<int> &dimensions) const {
        
        ResultSet result;

        if(dimensions.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        if(copy.back() >= rankOf())
            throw std::runtime_error("NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");

        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

        for (uint idx = 0; idx < tadPack.numberOfTads(); idx++ ) {
            auto array = new NDArray(bufferWithOffset(tadPack.primaryOffsets()[idx]), tadPack.primaryShapeInfo(), getWorkspace(), false, false);
            result.push_back(array);
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allTensorsAlongDimension(const std::initializer_list<int>& dimensions) const {
        return allTensorsAlongDimension(std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allTensorsAlongDimension(const std::vector<int> &dimensions) const {
        return new ResultSet(std::move(allTensorsAlongDims(dimensions)));
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allExamples() const {
        std::vector<int> dimensions(rankOf() - 1);
        for (int e = 1; e < rankOf(); e++)
            dimensions[e-1] = e;

        return allTensorsAlongDimension(dimensions);
    }

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::getOffset(const Nd4jLong i) const {

    if (i >= _length)
        throw std::invalid_argument("NDArray::getOffset: input index is out of array length !");    
    
    return shape::getIndexOffset(i, _shapeInfo, _length);
}

////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(IndicesList& idx) const {

        // auto xType = this->dataType();
        // // scalar subarray edge case
        // if (idx.isScalar()) {
        //     auto pnt = idx.at(0)->getIndices().at(0);
        //     //return new NDArray('c', {1, 1}, {this->e(pnt)});
        //     BUILD_SINGLE_SELECTOR(xType, return NDArrayFactory::create_, (this->e<float>(pnt), this->_workspace), LIBND4J_TYPES);
        // }

        const int idxSize = idx.size();
        if (idxSize != this->rankOf())
            throw std::runtime_error("Number of indices should match");

        std::vector<Nd4jLong> indexes(3 * idxSize);

        // convert IndicesList to vector
        for (int d = 0; d < idxSize; ++d) {

            if (idx.at(d)->isAll()) {
                indexes[3 * d]     = 0;                             // first
                indexes[3 * d + 1] = 0;                             // last
                indexes[3 * d + 2] = 1;                             // stride
            }
            else if (idx.at(d)->isPoint()) {
                indexes[3 * d]     = idx.at(d)->getIndices().at(0); // first
                indexes[3 * d + 1] = indexes[3 * d] + 1;            // last
                indexes[3 * d + 2] = 1;                             // stride
            }
            else if (idx.at(d)->isInterval()) {
                indexes[3 * d]     = idx.at(d)->getIndices().at(0); // first
                indexes[3 * d + 1] = idx.at(d)->getIndices().size();// last
                indexes[3 * d + 2] = idx.at(d)->stride();           // stride
            }
            else {
                indexes[3 * d]     = idx.at(d)->getIndices().at(0); // first
                indexes[3 * d + 1] = idx.at(d)->getIndices().at(1); // last
                indexes[3 * d + 2] = idx.at(d)->getIndices().at(2); // stride
            }
        }

        return new NDArray((*this)(indexes, true, true));
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(const std::initializer_list<NDIndex*>& idx) const {

        const int idxSize = idx.size();
        if (idxSize != this->rankOf())
            throw std::runtime_error("NDArray::subarray: number of indices should match the array rank");

        std::vector<Nd4jLong> indexes(3 * idxSize);

        // convert NDIndex to vector
        int d = 0;
        for (const auto& item : idx) {

            if (item->isAll()) {
                indexes[3 * d]     = 0;                             // first
                indexes[3 * d + 1] = 0;                             // last
                indexes[3 * d + 2] = 1;                             // stride
            }
            else if (item->isPoint()) {
                indexes[3 * d]     = item->getIndices().at(0);      // first
                indexes[3 * d + 1] = indexes[3 * d] + 1;            // last
                indexes[3 * d + 2] = 1;                             // stride
            }
            else if (item->isInterval()) {
                indexes[3 * d]     = item->getIndices().at(0);      // first
                indexes[3 * d + 1] = item->getIndices().size();     // last
                indexes[3 * d + 2] = item->stride();                // stride
            }
            else {
                indexes[3 * d]     = item->getIndices().at(0);      // first
                indexes[3 * d + 1] = item->getIndices().at(1);      // last
                indexes[3 * d + 2] = item->getIndices().at(2);      // stride
            }
            ++d;
        }

        // release NDIndices
        for (auto i: idx)
            delete i;

        return new NDArray((*this)(indexes, true, true));
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(const Intervals& idx) const {

        const int idxSize = idx.size();
        if (idxSize != this->rankOf())
            throw std::runtime_error("NDArray::subarray: number of indices should match the rank of array!");

        std::vector<Nd4jLong> indexes(2 * idxSize);

        // convert Intervals to vector
        for (int d = 0; d < idxSize; ++d) {

             if (idx[d].empty()) {
                indexes[2 * d]     = 0; // first
                indexes[2 * d + 1] = 0; // last
            }
            else {
                indexes[2 * d]     = idx[d][0]; // first
                indexes[2 * d + 1] = idx[d][1]; // last
            }
        }

        return new NDArray((*this)(indexes, true));
    }

    void NDArray::nullify() {
        if (isEmpty() || _buffer == nullptr)
            return;

        if (this->isView() || this->ews() != 1) {
            this->assign(0);
        } else {
            std::memset(_buffer, 0, this->lengthOf() * this->sizeOfT());
        }
    }

////////////////////////////////////////////////////////////////////////
void NDArray::getSubArrShapeAndOffsets(const std::vector<int>& dimsToExclude, Nd4jLong* &subArrShapeInfo, Nd4jLong* &subArrOffsets, bool keepUnitiesInShape) const {

    const int rank = rankOf();
    const int subArrRank = (rank == dimsToExclude.size() || keepUnitiesInShape) ? rank : rank - dimsToExclude.size();
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

    // allocate memory
    ALLOCATE(subArrShapeInfo, _workspace, shape::shapeInfoLength(subArrRank), Nd4jLong);
    ALLOCATE(subArrOffsets,   _workspace, numOfSubArrs, Nd4jLong);

    shape::calcSubArrShapeAndOffsets(_shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(), subArrShapeInfo, subArrOffsets, keepUnitiesInShape);
}

    //BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, Y value), LIBND4J_TYPES, LIBND4J_TYPES);
/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

