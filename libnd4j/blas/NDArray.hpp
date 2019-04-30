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

// $NDArray.hpp - architech-independent implementations (both cuda and cpu).
//
#ifndef __NDARRAY__HPP__
#define __NDARRAY__HPP__

#include <array/ShapeDescriptor.h>
#include <ConstantShapeHelper.h>
#include <ConstantShapeHelper.h>
#include <ConstantTadHelper.h>

namespace nd4j {

    template <>
    utf8string NDArray::e(const Nd4jLong i) const;
    template <>
    std::string NDArray::e(const Nd4jLong i) const;
    

    NDArray* NDArray::getView() {
        auto view = new NDArray();
        view->_isView = true;
        view->_shapeInfo = _shapeInfo;
        view->_buffer = _buffer;
        view->_context = _context;
        view->_isBuffAlloc = false;

        return view;
    }

    template <typename T>
    NDArray* NDArray::asT() const{
        auto result = new NDArray(ordering(), getShapeAsVector(), DataTypeUtils::fromT<T>());
        auto l = this->lengthOf();

        NativeOpExecutioner::execTransformAny(_context, transform::AnyOps::Assign, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, nullptr, nullptr, nullptr);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArray::asT, () const, LIBND4J_TYPES);

/////////////////   7///////////////////////////////////////////////////////
// move constructor
NDArray::NDArray(NDArray&& other) noexcept {

    _isView       = other._isView;
    _buffer       = other._buffer;
    _shapeInfo    = other._shapeInfo;
    _context      = other._context;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;

    _writeDevice = other._writeDevice;
    _readDevice = other._readDevice;
    _writeHost = other._writeHost;
    _readHost = other._readHost;
    _opCounter = other._opCounter;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;
    this->_length = other.lengthOf();
}

////////////////////////////////////////////////////////////////////////
//constructor, create empty array at given workspace
NDArray::NDArray(nd4j::graph::LaunchContext* context) {
    _buffer    = nullptr;
    _bufferD   = nullptr;
    _shapeInfo = nullptr;
    _shapeInfoD = nullptr;
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _context = context;
    _length = 0;
}

////////////////////////////////////////////////////////////////////////
// default constructor
 NDArray::NDArray() {
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
}

std::pair<bool, bool> NDArray::isShapeOwner() {
    std::pair<bool, bool> result(false, false);
    return result;
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros, set dtype as array type
NDArray::NDArray(Nd4jLong* shapeInfo, const bool copyStrides, nd4j::graph::LaunchContext* context): 
                NDArray(shapeInfo, ArrayOptions::dataType(shapeInfo), copyStrides, context) {
}   

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, graph::LaunchContext* context, const bool isBuffAlloc) {
    
    if (buffer == nullptr && ArrayOptions::arrayType(shapeInfo) != ArrayType::EMPTY)
        throw std::runtime_error("NDArray constructor: can't be initalized with nullptr buffer !");
    
    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo !");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");

    _context = context;
    _isAttached = _context->getWorkspace() != nullptr;

    setShapeInfo(shapeInfo);

    if (this->isEmpty()) {        
        tickReadDevice();
        tickReadHost();
    }
    else {        
        _buffer = reinterpret_cast<int8_t *>(buffer);            
        triggerAllocationFlag(isBuffAlloc);
        #ifdef __CUDABLAS__
            ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
            cudaMemcpy(_bufferD, _buffer, _length * sizeOfT(), cudaMemcpyHostToDevice);
            _isBuffDAlloc = true;
            tickWriteDevice();
            tickReadHost();
        #endif
    }
}

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
// we suppose the content of both (device and host) buffers is identical 
NDArray::NDArray(void *buffer, void* bufferD, Nd4jLong *shapeInfo, graph::LaunchContext* context, const bool isBuffAlloc, const bool isBuffDAlloc) {

    if (buffer == nullptr && bufferD == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized with both nullptr buffers !");

    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor cuda: can't be initalized without shapeinfo");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("NDArray constructor cuda: rank of NDArray can't exceed 32");
     
    _context = context;

    setShapeInfo(shapeInfo);
    
    if (!isEmpty()) {
        
         if(buffer != nullptr) {
            _buffer = reinterpret_cast<int8_t *>(buffer);        
            triggerAllocationFlag(isBuffAlloc);
            tickReadHost();
        }        
        if(bufferD != nullptr) {
            _bufferD = reinterpret_cast<int8_t *>(bufferD);
            _isBuffDAlloc = isBuffDAlloc;
            tickReadDevice();
        }
    }
}

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

        for (int e = 0; e < _length; e++)
            vector[e] = this->e<T>(e);

        return vector;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::getBufferAsVector(), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
    std::vector<int64_t> NDArray::getShapeAsFlatVector() {
        std::vector<int64_t> vector(this->rankOf());

        for (int e = 0; e < this->rankOf(); e++)
            vector[e] = static_cast<int64_t>(this->sizeAt(e));

        return vector;
    }


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
    std::vector<int8_t> NDArray::asByteVector() {
        std::vector<int8_t> result((unsigned long long) this->lengthOf() * sizeOfT());
        lazyAllocateBuffer();
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
    void* NDArray::getSpecialBuffer() const{
        if (_bufferD == nullptr)
            return _buffer;

        // FIXME: this should be fixed once CUDA backend added
        return _bufferD;
    }

////////////////////////////////////////////////////////////////////////
    Nd4jLong* NDArray::getSpecialShapeInfo() const{
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
// move assignment operator
NDArray& NDArray::operator=(NDArray&& other) noexcept {

    if (this == &other) 
        return *this;

    if(_context->getWorkspace() == nullptr) {
            
        if(_isBuffAlloc) delete []_buffer;

        if(_isBuffDAlloc)  RELEASE_SPECIAL(_bufferD, nullptr);
    }

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _context      = other._context;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;
    _length       = other._length;
    _writeDevice = other._writeDevice;
    _readDevice = other._readDevice;
    _writeHost = other._writeHost;
    _readHost = other._readHost;
    _opCounter = other._opCounter;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;

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
    template NDArray& NDArray::operator=(const bfloat16 scalar);
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
        if (_isBuffAlloc && _context->getWorkspace() == nullptr)
            delete[] _buffer;
    }
}

// This method assigns values of given NDArray to this one, wrt order
    void NDArray::assign(const NDArray *other) {
        assign(*other);
    }

// This method assigns given value to all elements in this NDArray
    void NDArray::assign(const double value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(this->_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp.specialBuffer(), temp.specialShapeInfo(), nullptr);
    }

    void NDArray::assign(const float value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(this->_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const float16 value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(value, this->_context);
        NativeOpExecutioner::execScalar(this->_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const bfloat16& value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const Nd4jLong value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const int value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const int16_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const uint8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const int8_t value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
    }

    void NDArray::assign(const bool value) {
        // just fire scalar
        auto temp = NDArrayFactory::create(this->dataType(), value, this->_context);
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, temp.buffer(), temp.shapeInfo(), temp._bufferD, temp._shapeInfoD, nullptr);
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
        result->triggerAllocationFlag(true);
        result->setContext(nd4j::graph::LaunchContext::defaultContext());

//        auto d1 = this->dataType();
  //      auto d2 = result->dataType();

   //     auto s1 = this->sizeOfT();
   //     auto s2 = result->sizeOfT();

        result->assign(*this);

        return result;
    }

    NDArray NDArray::varianceNumber(nd4j::variance::Ops op, bool biasCorrected) {
        NDArray res(DataTypeUtils::pickFloatingType(dataType()), _context);
        if (!isActualOnDeviceSide())
            syncToDevice();

        res.tickWriteDevice();
        NativeOpExecutioner::execSummaryStatsScalar(_context, op,this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo(), biasCorrected);
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    NDArray NDArray::sumNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::sumNumber: you can't use this method on String array!");
        NDArray res(_dataType, _context);
        NDArray::prepareSpecialUse({&res}, {this});

        NativeOpExecutioner::execReduceSameScalar(_context, nd4j::reduce::SameOps::Sum, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    NDArray NDArray::meanNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::meanNumber: you can't use this method on String array!");
        NDArray res(DataTypeUtils::pickFloatingType(dataType()), _context);
        NDArray::prepareSpecialUse({&res}, {this});

        NativeOpExecutioner::execReduceFloatScalar(_context, nd4j::reduce::FloatOps::Mean, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
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


    void NDArray::setContext(nd4j::graph::LaunchContext *context) {

        this->_context= context;
        if (_context == nullptr)
            _context = nd4j::graph::LaunchContext::defaultContext(); // empty context for default cases
    }

    void* NDArray::bufferWithOffset(Nd4jLong offset) const {
        
        return _buffer != nullptr ? _buffer + (offset * sizeOfT()) : nullptr;
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
    
    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(!isR())
        ArrayOptions::setDataType(newShape, Environment::getInstance()->defaultFloatDataType());

    NDArray result(newShape, true, _context);
    if (!isActualOnDeviceSide())
        syncToDevice();

    result.tickWriteDevice();

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());

    NDArray result(newShape, true, _context);
    if (!isActualOnDeviceSide())
        syncToDevice();

    result.tickWriteDevice();

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    RELEASE(newShape, _context->getWorkspace());
    
    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(!isB())
        ArrayOptions::setDataType(newShape, DataType::BOOL);

    NDArray result(newShape, true, _context);
    if (!isActualOnDeviceSide())
        syncToDevice();

    result.tickWriteDevice();

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    
    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(_dataType != DataType::INT64)
        ArrayOptions::setDataType(newShape, DataType::INT64);

    NDArray result(newShape, true, _context);
    if (!isActualOnDeviceSide())
        syncToDevice();

    result.tickWriteDevice();

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
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

        auto shape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());

        if (!isActualOnDeviceSide())
            syncToDevice();

        result.tickWriteDevice();

        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");

        NDArray result(_dataType, _context);
        if (!isActualOnDeviceSide())
            syncToDevice();

        result.tickWriteDevice();

        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::BOOL, this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());
        if (!isActualOnDeviceSide())
            syncToDevice();

        result.tickWriteDevice();

        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::LongOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::INT64, this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());

        if (!isActualOnDeviceSide())
            syncToDevice();

        result.tickWriteDevice();

        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::reduceNumber(nd4j::reduce::FloatOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataTypeUtils::pickFloatingType(_dataType))
            throw std::invalid_argument("NDArray::reduceNumber FloatOps: target array should be scalar and have corresponding float type!");
        if (!isActualOnDeviceSide())
            syncToDevice();

        target.tickWriteDevice();

        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }

    void NDArray::reduceNumber(nd4j::reduce::SameOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != _dataType)
            throw std::invalid_argument("NDArray::reduceNumber SameOps: target array should be scalar and have same type as this array!");

        if (!isActualOnDeviceSide())
            syncToDevice();

        target.tickWriteDevice();

        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target._buffer, target._shapeInfo, target._bufferD, target._shapeInfoD);
    }

    void NDArray::reduceNumber(nd4j::reduce::BoolOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::reduceNumber BoolOps: target array should be scalar and have bool type!");

        if (!isActualOnDeviceSide())
            syncToDevice();

        target.tickWriteDevice();

        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target._buffer, target._shapeInfo, target._bufferD, target._shapeInfoD);
    }

    void NDArray::reduceNumber(nd4j::reduce::LongOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::INT64)
            throw std::invalid_argument("NDArray::reduceNumber LongOps: target array should be scalar and have long type!");

        if (!isActualOnDeviceSide())
            syncToDevice();

        target.tickWriteDevice();

        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }

//////////////////////////////////////////////////////////////////////////
    NDArray NDArray::indexReduceNumber(nd4j::indexreduce::Ops op, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::indexReduceNumber: you can't use this method on String array!");

        auto res = NDArrayFactory::create<Nd4jLong>(0);
        NDArray::prepareSpecialUse({&res}, {this});

        NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams == nullptr ? nullptr : extraParams->argumentsAsT(this->dataType()), res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
        return res;
    }


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
        if (!isActualOnHostSide())
            syncToHost();
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
                if (this->dataType() != nd4j::DataType::INT64 && this->dataType() != nd4j::DataType::UINT64)
                    printf("%d", this->e<int>(e));
                else
                    printf("%llu", this->e<Nd4jLong>(e));
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
        
        if (arr->rankOf() == 1) {            
            printf("[ ");
            for (Nd4jLong i = 0; i < arr->lengthOf(); ++i) {                
                if (arr->isR())
                    printf("%f, ", arr->e<float>(i));
                else if (arr->isZ())
                    printf("%lld, ", arr->e<Nd4jLong>(i));
                else if (arr->isB())
                    printf("%s, ", arr->e<bool>(i)?"true":"false");
                else if (arr->isS()) 
                    printf("\"%s\", ", arr->e<std::string>(i).c_str());
            }
            printf("]\n");
        }
        else if (arr->rankOf() == 2) {
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
                        printf("%f,", arr->e<float>(row, col));
                    else if (arr->isZ())
                        printf("%lld,", arr->e<Nd4jLong>(row, col));
                    else if (arr->isB())
                        printf("%s,", arr->e<bool>(row, col)?"true":"false");
                    else if (arr->isS()) 
                        printf("\"%s\",", arr->e<std::string>(row * cols + col).c_str());                    
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

        if (!isActualOnHostSide())
            syncToHost();

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
        else if (rowFlag && ews()==1)
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

    template <typename T>
    void* NDArray::templatedPointerShift(const Nd4jLong offset) const {
        return reinterpret_cast<T*>(_buffer) + offset;
    }
    BUILD_SINGLE_TEMPLATE(template void* NDArray::templatedPointerShift, (const Nd4jLong offset) const, LIBND4J_TYPES);

// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected
    NDArray* NDArray::transpose() const {
        
        Nd4jLong* newShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, true, _context->getWorkspace());
        auto newArr = new NDArray(_buffer, _bufferD, newShapeInfo, _context, false, false);
        newArr->transposei();
        RELEASE(newShapeInfo, _context->getWorkspace());

        return newArr;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transp() const {
    int shapeInfoLength = shape::shapeInfoLength(rankOf());
    Nd4jLong* newShapeInfo;

    ALLOCATE(newShapeInfo , _context->getWorkspace(), shapeInfoLength, Nd4jLong);
    memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

    NDArray newArr(_buffer, newShapeInfo, _context, false);

    newArr.transposei();

    return newArr;
}


////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected 
    void NDArray::transpose(NDArray& target) const {
        
        auto correctShape = ShapeUtils::evalTranspShapeInfo(*this, _context->getWorkspace());
        if(!shape::equalsStrict(correctShape, target.getShapeInfo()))
            throw std::runtime_error("NDArray::transpose method: the shapeInfo of target array is wrong !");

    // check whether target has allocated (its own) buffer
    if (target._isBuffAlloc) 
        RELEASE(reinterpret_cast<int8_t *>(target._buffer), target._context->getWorkspace());

    target._buffer = _buffer;
    // don't forget to indicate that memory for new array was allocated
    target.triggerAllocationFlag(false);
    target._isView = true;

    RELEASE(correctShape, _context->getWorkspace());
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
        ALLOCATE(ptr, _context->getWorkspace(), _length * DataTypeUtils::sizeOfElement(dtype), int8_t);

        // FIXME: if we're going to merge this - move loop into selector
        for (int e = 0; e < this->lengthOf(); e++) {
            BUILD_DOUBLE_SELECTOR(dtype, this->dataType(), templatedDoubleAssign, (ptr, e, this->_buffer, e), LIBND4J_TYPES, LIBND4J_TYPES);
        }
        return ptr;
    }


    void NDArray::setAttached(bool reallyAttached) {
        _isAttached = reallyAttached;
    };


//////////////////////////////////////////////////////////////////////////
    // calculate strides
    void NDArray::updateStrides(const char order) {
    	shape::updateStrides(_shapeInfo, order);
    	syncShape();
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
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(dimensions.size()), Nd4jLong);

        char order = o == 'a' ? this->ordering() : o;

        if (order == 'c')
            shape::shapeBuffer(dimensions.size(), dataType(), dimensions.data(), newShape);
        else
            shape::shapeBufferFortran(dimensions.size(), dataType(), dimensions.data(), newShape);

        setShapeInfo(newShape);

        RELEASE(newShape, _context->getWorkspace());
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

        Nd4jLong* newShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, true, _context->getWorkspace());                    
        auto newArr = new NDArray(_buffer, _bufferD, newShapeInfo, _context, false, false);
        newArr->reshapei(order, shape);

        RELEASE(newShapeInfo, _context->getWorkspace());
        return newArr;
    }

    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    void NDArray::tilei(const std::vector<Nd4jLong>& reps) {

        *this = this->tile(reps);
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
        auto shapeInfoPermuted = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
        // create array to be returned        
        auto ret = new NDArray(_buffer, _bufferD, shapeInfoPermuted, _context, false, false);
	    ret->_isView = true;

	    RELEASE(shapeInfoPermuted, _context->getWorkspace());
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
        auto data = dimensions.data();
        auto size = dimensions.size();
        return permute(data, size);
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
            RELEASE(target._buffer, target._context->getWorkspace());

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._context->getWorkspace());

        target._buffer = _buffer;
        target.setShapeInfo(shapeInfoNew);
        target.triggerAllocationFlag(false);
    }

    void NDArray::permute(const Nd4jLong *dimensions, const int rank, NDArray& target) const {
        if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
            throw std::runtime_error("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

        // check whether target has allocated (its own) buffer
        if (target._isBuffAlloc)
            RELEASE(target._buffer, target._context->getWorkspace());

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._context->getWorkspace());

        target._buffer = _buffer;
        target.setShapeInfo(shapeInfoNew);
        target.triggerAllocationFlag(false);
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::permute(const std::vector<int>& dimensions, NDArray& target) const {
        permute(dimensions.data(), dimensions.size(), target);
    }

    void NDArray::permute(const std::vector<Nd4jLong>& dimensions, NDArray& target) const {
        permute(dimensions.data(), dimensions.size(), target);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, const std::initializer_list<int> dimensions, const NDArray* tadArray, NDArray* target, ExtraArguments* extraArgs) {
        std::vector<int> vec(dimensions);
        applyBroadcast(op, vec, tadArray, target, extraArgs);
    }


    //////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, ExtraArguments *extraArgs) const {

        return new NDArray(this->applyTrueBroadcast(op, *other, extraArgs));
    }

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
        if(rankOf() != 2 || rows() != columns())
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
    template <>
    std::string* NDArray::bufferAsT() const {
        throw std::runtime_error("This method is NOT supposed to be used");
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    T * NDArray::bufferAsT() const {
        if (isS())
            throw std::runtime_error("You can't use this method on String array");

        lazyAllocateBuffer();
        if (!isActualOnHostSide())
            syncToHost();

        return reinterpret_cast<T*>(_buffer);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template, * NDArray::bufferAsT() const, LIBND4J_TYPES);



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
            throw std::runtime_error("NDArray::subarray: number of indices should match");

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


    NDArray* NDArray::asT(DataType dtype) const {
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
    NDArray* NDArray::cast(DataType dtype) const {
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



    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
            return varianceAlongDimension(op, biasCorrected, std::vector<int>(dimensions));
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op,const NDArray *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension(op, target, biasCorrected, std::vector<int>(dimensions));
    }
       
    ////////////////////////////////////////////////////////////////////////
    // addition operator array + array
    NDArray NDArray::operator+(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator+: you can't use this method on String array!");

        if (!Environment::getInstance()->isExperimentalBuild() && this->_dataType != other._dataType && (other._dataType != DataType::BOOL) ) {
            throw datatype_exception::build("NDArray::operator+: cannot add different types.", _dataType, other._dataType);
        }
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_context);
            NDArray::prepareSpecialUse({&result}, {this, &other});

            NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr);

            NDArray::registerSpecialUse({&result}, {this, &other});
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

    auto tmp = NDArrayFactory::create(_dataType, scalar, _context);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _context);
    NDArray::prepareSpecialUse({&result}, {this, &tmp});

    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {this, &tmp});
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

    auto tmp = NDArrayFactory::create(this->dataType(), scalar, this->_context);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _context);
    NDArray::prepareSpecialUse({&result}, {this, &tmp});

    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Subtract, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {this, &tmp});
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

    auto tmp = NDArrayFactory::create(this->dataType(), scalar, _context);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _context);
    NDArray::prepareSpecialUse({&result}, {this, &tmp});

    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {this, &tmp});
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

    auto tmp = NDArrayFactory::create(this->dataType(), scalar, _context);
    NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_dataType, DataTypeUtils::fromT<T>()), false, _context);
    NDArray::prepareSpecialUse({&result}, {this, &tmp});

    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Divide, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {this, &tmp});
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

    auto tmp = NDArrayFactory::create(scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float16>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator-(const bfloat16& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<bfloat16>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator-(const float& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator-(const double& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator-(const Nd4jLong& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<Nd4jLong>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator-(const int& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getContext());
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator/(const bfloat16& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<bfloat16>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}
////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator/(const float16& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float16>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator/(const float& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator/(const double& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

ND4J_EXPORT NDArray operator/(const int& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getContext());
    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});

    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({&result}, {&arr, &tmp});
    return result;
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator+=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");
    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
        throw nd4j::datatype_exception::build("NDArray operator+=: Cannot add different types", this->dataType(), other.dataType());
    }
    if (!this->isScalar() && other.isScalar()) {
        NativeOpExecutioner::execScalar(getContext(), nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExecutioner::execPairwiseTransform(this->getContext(), nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_context->getWorkspace()))
            throw std::invalid_argument("NDArray::operator+=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _context);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_context->getWorkspace());
    }
    tickWriteDevice();
}

void NDArray::operator-=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator-=: you can't use this method on String array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
        throw nd4j::datatype_exception::build("NDArray operator-=: Cannot subtract different types", this->dataType(), other.dataType());
    }

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_context->getWorkspace()))
            throw std::invalid_argument("NDArray::operator-=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _context);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_context);
    }
    tickWriteDevice();
}

void NDArray::operator*=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
        throw nd4j::datatype_exception::build("NDArray operator*=: Cannot multiply different types", this->dataType(), other.dataType());
    }

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
        this->tickWriteHost();
        this->tickWriteDevice();
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, nullptr);
        this->tickWriteHost();
        this->tickWriteDevice();
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_context->getWorkspace()))
            throw std::invalid_argument("NDArray::operator*=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _context);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_context->getWorkspace());
    }
}

void NDArray::operator/=(const NDArray& other) {
    if (isS() || other.isS())
        throw std::runtime_error("NDArray::operator/=: you can't use this method on String array!");
    if (other.isB())
        throw std::runtime_error("NDArray::operator/=: you can't divide by bool array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType()) {
        throw nd4j::datatype_exception::build("NDArray operator/=: Cannot divide different types", this->dataType(), other.dataType());
    }

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Divide, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, nullptr);
    }
    else{
        Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, this->_context->getWorkspace()))
            throw std::invalid_argument("NDArray::operator/=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), &other, this, false);
        }
        else {
            NDArray result(bShape, true, _context);
            this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), &other, &result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
        RELEASE(bShape, this->_context->getWorkspace());
    }
}
////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::operator+=(const T value) {
    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");

    auto other = NDArrayFactory::create(this->dataType(), value, _context);
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Add, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
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

    auto other = NDArrayFactory::create(this->dataType(), value, _context);
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Subtract, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
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

    auto other = NDArrayFactory::create(this->dataType(), scalar, _context);
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
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

    auto other = NDArrayFactory::create(this->dataType(), scalar, _context);
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Divide, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
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

        if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
            throw nd4j::datatype_exception::build("NDArray operator-: Cannot subtract different types", this->dataType(), other.dataType());
        }

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_context);
            NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Subtract, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // negative operator, it makes all array elements = -elements
    NDArray NDArray::operator-() const {
        if (isS())
            throw std::runtime_error("NDArray::negative-: you can't use this method on String array!");
        NDArray result(this->_shapeInfo, false, this->_context);
        NativeOpExecutioner::execTransformSame(_context, nd4j::transform::Neg, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr, nullptr, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*array
    NDArray NDArray::operator*(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator*: you can't use this method on String array!");

        if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
            throw nd4j::datatype_exception::build("NDArray operator*: Cannot multiply different types", this->dataType(), other.dataType());
        }
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_context);
            NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array/array
    NDArray NDArray::operator/(const NDArray& other) const {
        if (isS())
            throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
        if (other.isB())
            throw std::runtime_error("NDArray::operator/: you can't divide by bool array!");
        if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType()) {
            throw nd4j::datatype_exception::build("NDArray operator/: Cannot divide different types", this->dataType(), other.dataType());
        }

        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_context);
            NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Divide, _buffer, _shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr);
            return result;
        }

        return this->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Divide(), other);
    }

    ////////////////////////////////////////////////////////////////////////
    // mathematical multiplication of two arrays
    NDArray mmul(const NDArray& left, const NDArray& right) {
        if (left.isS() || right.isS())
            throw std::runtime_error("mmul friend function: you can't use this function on String array!");
        auto ptr = MmulHelper::mmul(const_cast<NDArray*>(&left), const_cast<NDArray*>(&right), nullptr, 1., 0.);
        NDArray result(std::move(*ptr));
        delete ptr;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    /*
    void NDArray::assign(const NDArray& other, const Intervals& idx) {
        auto subarr = this->subarray(idx);
        subarr->assign(&other);
        delete subarr;
    }
     */

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

        NDArray result(const_cast<Nd4jLong*>(shapeInfo), false, _context);
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

#pragma omp parallel for reduction(sumT:sum) if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i = 0; i < minDim; ++i)
            sum += e<double>(i*offset);

        return sum;
    }


////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::dupUninitialized() const {
        Nd4jLong* newShape(nullptr);
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(_shapeInfo), Nd4jLong);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(_shapeInfo));

        int8_t* buffer(nullptr);
        ALLOCATE(buffer, _context->getWorkspace(), _length * sizeOfT(), int8_t);
        auto result = new NDArray(buffer, newShape, _context);
        result->triggerAllocationFlag(true);        

        return result;
    }

    NDArray NDArray::quantize(NDArray &array) {
        return *(quantize(&array));
    }

    NDArray* NDArray::quantize(NDArray *array) {

        if(array->isR())
            throw std::invalid_argument("NDArray::quantize: type of array should be from real space!");

        auto ws = array->getContext()->getWorkspace();
        char *buffer = nullptr;
        Nd4jLong *shapeInfo;

        // allocate buffers
        ALLOCATE(buffer, ws, TypeCast::estimateQuantizedSize(array->lengthOf()), char);
        COPY_SHAPE_EX(array->shapeInfo(), shapeInfo, ws);

        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_QUANTIZED);

        auto result = new NDArray(buffer, shapeInfo, array->getContext());
        result->triggerAllocationFlag(true);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::multipleTensorsAlongDimension(const std::vector<int> &indices, const std::vector<int> &dimensions) const {
        auto result = new ResultSet();

        if (indices.size() == 0)
            return result;

        auto pack = ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, const_cast<int*>(dimensions.data()), dimensions.size());

        auto tadLength = shape::length(pack.primaryShapeInfo());
        auto numTads = _length / tadLength;

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("NDArray::multipleTensorsAlongDimension: index %i is higher then number of TADs: %i\n", idx, numTads);
                throw std::runtime_error("Bad index");
            }

            auto array = new NDArray(bufferWithOffset(pack.primaryOffsets()[idx]), specialBufferWithOffset(pack.primaryOffsets()[idx]), pack.primaryShapeInfo());
            result->push_back(array);
        }

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

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::getOffset(const Nd4jLong i) const {

    if (i >= _length)
        throw std::invalid_argument("NDArray::getOffset: input index is out of array length !");    

    if(ordering() == 'c') {
        if(ews() == 1)
            return i;
        if(ews() > 1)
            return i * ews();
    }

    return shape::getIndexOffset(i, _shapeInfo, _length);
}

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::diagonal(const char type) const {

        if (isS())
            throw std::runtime_error("NDArray::diagonal: you can't use this method on String array!");

        const char order = ordering();
        const int  rank  = rankOf();
        Nd4jLong *outShapeInfo;
        ALLOCATE(outShapeInfo, _context->getWorkspace(), 8, Nd4jLong);
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
            outShapeInfo[6] = 0;
        }

        ArrayOptions::setDataType(outShapeInfo, this->dataType());

        auto result = new NDArray(_buffer, outShapeInfo, _context, false);
        result->_bufferD = _bufferD;
        result->_isBuffDAlloc = false;

        if (isActualOnDeviceSide())
            result->tickWriteDevice();
        else if (isActualOnHostSide())
            result->tickWriteHost();

        return result;
    }

////////////////////////////////////////////////////////////////////////
    ResultSet* NDArray::allTensorsAlongDimension(const std::vector<int> &dimensions) const {
        auto result = new ResultSet();

        if(dimensions.size() == 0)
            return result;

        if(dimensions.back() >= rankOf())
            throw std::runtime_error("NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");


        auto pack = ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, const_cast<int*>(dimensions.data()), dimensions.size());
        auto numTads = _length / shape::length(pack.primaryShapeInfo());

        for (int idx = 0; idx < numTads; idx++ ) {
            // FIXME: why is this? why are we allocating this?
            lazyAllocateBuffer();
            makeBothBuffersActual();
            auto array = new NDArray(bufferWithOffset(pack.primaryOffsets()[idx]), specialBufferWithOffset(pack.primaryOffsets()[idx]), pack.primaryShapeInfo(), _context, false, false);

            result->push_back(array);
        }

        return result;
    }

////////////////////////////////////////////////////////////////////////
// operator returns sub-array with buffer pointing at this->_buffer + certain offset
NDArray NDArray::operator()(const std::vector<Nd4jLong>& idx, const bool keepUnitiesInShape, const bool isStrided)  const {
    
    const int rank = rankOf();
    Nd4jLong *newShape;
    ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
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
    shape::calcEws(newShape, subArrLen);

    #ifdef __CUDABLAS__
        makeBothBuffersActual();
        NDArray result(bufferWithOffset(offset), specialBufferWithOffset(offset), newShape, _context, false, false);
    #else            
        NDArray result(bufferWithOffset(offset), newShape, _context, false);
    #endif     
    
    if(!keepUnitiesInShape) {
        const int coeff = isStrided ? 3 : 2;
        std::vector<Nd4jLong> nonUnitDims;
        
        for (int d = 0; d < rank; ++d) 
            if(!(idx[coeff*d] != idx[coeff*d+1] && newShape[d+1] == 1))
                nonUnitDims.push_back(newShape[d+1]);        

        if(nonUnitDims.size() != rank)
            result.reshapei(nonUnitDims);
    }

    RELEASE(newShape, _context->getWorkspace());

    return result;
}


////////////////////////////////////////////////////////////////////////
NDArray NDArray::operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape)  const {
    std::vector<Nd4jLong> idxRanges(2 * rankOf());
        
    ShapeUtils::evalIdxRangesForSubArr(subArrIdx, _shapeInfo, dimsToExclude, idxRanges.data());

    return (*this)(idxRanges, keepUnitiesInShape);
}
  
////////////////////////////////////////////////////////////////////////
void NDArray::getSubArrShapeAndOffsets(const std::vector<int>& dimsToExclude, Nd4jLong* &subArrShapeInfo, Nd4jLong* &subArrOffsets, bool keepUnitiesInShape) const {

    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);
    const int rank = rankOf();        
    const int dimsSize = dimsToExclude.size();        

    // allocate memory 
    ALLOCATE(subArrShapeInfo, _context->getWorkspace(), shape::shapeInfoLength(rank - dimsSize), Nd4jLong);
    ALLOCATE(subArrOffsets,   _context->getWorkspace(), numOfSubArrs, Nd4jLong);        

    Nd4jLong *outShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, true, _context->getWorkspace());        
    std::vector<Nd4jLong> shape(dimsSize), strides(dimsSize);

    Nd4jLong subArrLen = 1;

    for(int j = dimsSize - 1, i = rank - 1; i >= 0; --i) {
        if(j >= 0 && i == dimsToExclude[j]) {
            strides[j] = shape::stride(outShapeInfo)[i];
            shape[j--] = shape::shapeOf(outShapeInfo)[i];
            shape::shapeOf(outShapeInfo)[i] = 1;
        }
        else
            subArrLen *= shape::shapeOf(outShapeInfo)[i];
    }

    // evaluate ews
    shape::calcEws(outShapeInfo, subArrLen);

    // calculation of sub-array offsets (subArrOffsets)
    shape::calcSubArrOffsets(numOfSubArrs, dimsSize, shape.data(), strides.data(), subArrOffsets);
        
    // remove unities from outShapeInfo if required 
    if(!keepUnitiesInShape) {
        std::vector<Nd4jLong> shapeNoUnities = ShapeUtils::evalDimsWithoutUnities(outShapeInfo);
        //shape::reshapeCF(rank, outShapeInfo, shapeNoUnities.size(), shapeNoUnities.data(), ordering() == 'f', subArrShapeInfo);
        throw std::runtime_error("FIXME");
    }
    else
        memcpy(subArrShapeInfo, outShapeInfo, shape::shapeInfoLength(rank)*sizeof(Nd4jLong));
        
    RELEASE(outShapeInfo, _context->getWorkspace());
}


//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const Nd4jLong *shapeInfo) {
        
    if (shapeInfo != nullptr) {

        ShapeDescriptor descriptor(shapeInfo);
        auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);
            
        _shapeInfo  = reinterpret_cast<Nd4jLong *>(shapeBuffer.primary());
        #ifdef __CUDABLAS__
            _shapeInfoD = reinterpret_cast<Nd4jLong *>(shapeBuffer.special());
        #endif          

        if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
            _length = 0;
        else
            _length = shape::length(_shapeInfo);

        _dataType = ArrayOptions::dataType(_shapeInfo);
    }
    else {
        _dataType = nd4j::DataType::INHERIT;
        _shapeInfoD = _shapeInfo = nullptr;        
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const Nd4jLong *shapeInfo, const nd4j::DataType dtype) {

    if (shapeInfo != nullptr) {

        Nd4jLong* shapeInfoTemp = ShapeBuilders::copyShapeInfoAndType(shapeInfo, dtype, true, _context->getWorkspace());
        ShapeDescriptor descriptor(shapeInfoTemp);
        auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

        _shapeInfo  = reinterpret_cast<Nd4jLong *>(shapeBuffer.primary());
        #ifdef __CUDABLAS__
            _shapeInfoD = reinterpret_cast<Nd4jLong *>(shapeBuffer.special());
        #endif

        if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
            _length = 0;
        else
            _length = shape::length(_shapeInfo);        

        _dataType = dtype;
    } 
    else {
        _dataType = nd4j::DataType::INHERIT;            
        _shapeInfoD = _shapeInfo = nullptr;
    }
}


//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const ShapeDescriptor& descriptor) {

    auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(const_cast<ShapeDescriptor &>(descriptor));

    _shapeInfo  = reinterpret_cast<Nd4jLong *>(shapeBuffer.primary());
    #ifdef __CUDABLAS__
        _shapeInfoD = reinterpret_cast<Nd4jLong *>(shapeBuffer.special());
    #endif

    if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
        _length = 0;
    else
        _length = shape::length(_shapeInfo);

    _dataType = ArrayOptions::dataType(_shapeInfo);
}

/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

