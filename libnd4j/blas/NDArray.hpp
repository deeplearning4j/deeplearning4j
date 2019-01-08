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
        view->_isShapeAlloc = false;
        view->_isBuffAlloc = false;

        return view;
    }

    template <typename T>
    NDArray* NDArray::asT() {
        auto result = new NDArray(ordering(), getShapeAsVector(), DataTypeUtils::fromT<T>());
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
// move constructor
NDArray::NDArray(NDArray&& other) noexcept {

    _isView       = other._isView;
    _buffer       = other._buffer;
    _shapeInfo    = other._shapeInfo;
    _context      = other._context;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
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
    _isShapeAlloc = false;
    _context = context;
    _length = 0;
}

////////////////////////////////////////////////////////////////////////
// default constructor
 NDArray::NDArray() {

    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros, set dtype as array type
NDArray::NDArray(Nd4jLong* shapeInfo, const nd4j::DataType dtype, const bool copyStrides, nd4j::graph::LaunchContext* context, const bool isShapeAlloc): 
                NDArray(shapeInfo, copyStrides, context, isShapeAlloc) {
    
    ArrayOptions::setDataType(_shapeInfo, _dataType);
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
        if (_buffer == nullptr)
            lazyAllocateBuffer();
        for (Nd4jLong e = 0; e < numElements; e++) {
            this->p(e, start + (step * e));
        }
        syncToDevice();
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

    if(_isBuffAlloc && _context->getWorkspace() == nullptr)
        delete []_buffer;
    if(_isShapeAlloc && _context->getWorkspace() == nullptr)
        delete []_shapeInfo;

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _context      = other._context;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
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
        if (_isShapeAlloc && _context->getWorkspace() == nullptr)
            delete[] _shapeInfo;

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
        result->_isBuffAlloc = true;
        result->_isShapeAlloc = true;
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
        NativeOpExecutioner::execSummaryStatsScalar(_context, op,this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo(), biasCorrected);
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    NDArray NDArray::sumNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::sumNumber: you can't use this method on String array!");
        NDArray res(_dataType, _context);
        NativeOpExecutioner::execReduceSameScalar(_context, nd4j::reduce::SameOps::Sum, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
        return res;
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    NDArray NDArray::meanNumber() const {
        if (isS())
            throw std::runtime_error("NDArray::meanNumber: you can't use this method on String array!");
        NDArray res(DataTypeUtils::pickFloatingType(dataType()), _context);
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
    
    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(!isR())
        ArrayOptions::setDataType(newShape, Environment::getInstance()->defaultFloatDataType());

    NDArray result(newShape, true, _context, true);    

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());

    NDArray result(newShape, true, _context, true);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(!isB())
        ArrayOptions::setDataType(newShape, DataType::BOOL);

    NDArray result(newShape, true, _context, true);

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDims(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    
    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
    if(_dataType != DataType::INT64)
        ArrayOptions::setDataType(newShape, DataType::INT64);

    NDArray result(newShape, true, _context, true);    

    reduceAlongDimension(op, &result, copy, keepDims, supportOldShapes, false);

    return result;
}


//////////////////////////////////////////////////////////////////////////
void NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != _dataType)
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: requires target array to be present and have same dtype as input");

        std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension SameOps: wrong target shape!");
        RELEASE(newShape, _context->getWorkspace());
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExecutioner::execReduceSameScalar(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());
    else {
        shape::TAD tad(_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NativeOpExecutioner::execReduceSame(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
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
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension BoolOps: wrong target shape!");
        RELEASE(newShape, _context->getWorkspace());
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExecutioner::execReduceBoolScalar(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());
    else {
        shape::TAD tad(_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NativeOpExecutioner::execReduceBool(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
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
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension LongOps: wrong target shape!");
        RELEASE(newShape, _context->getWorkspace());
    }

    if(rankOf() == copy.size() || copy.empty())
        //target->_buffer[0] = functions::reduce::ReduceFloatFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        NativeOpExecutioner::execReduceLongScalar(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());
    else {
        shape::TAD tad(_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NativeOpExecutioner::execReduceLong(_context, op, this->getBuffer(), this->getShapeInfo(), this->getSpecialBuffer(), this->getSpecialShapeInfo(), nullptr, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), copy.data(), copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
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

        auto shape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());

        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");

        NDArray result(_dataType, _context);
        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::BOOL, this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());

        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

    NDArray NDArray::reduceNumber(nd4j::reduce::LongOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");

        auto shape = ShapeBuilders::createScalarShapeInfo(DataType::INT64, this->_context->getWorkspace());
        NDArray result(shape, true, this->_context);
        RELEASE(shape, this->_context->getWorkspace());

        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
        return result;
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::reduceNumber(nd4j::reduce::FloatOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataTypeUtils::pickFloatingType(_dataType))
            throw std::invalid_argument("NDArray::reduceNumber FloatOps: target array should be scalar and have corresponding float type!");

        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }

    void NDArray::reduceNumber(nd4j::reduce::SameOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != _dataType)
            throw std::invalid_argument("NDArray::reduceNumber SameOps: target array should be scalar and have same type as this array!");

        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target._buffer, target._shapeInfo, target._bufferD, target._shapeInfoD);
    }

    void NDArray::reduceNumber(nd4j::reduce::BoolOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::reduceNumber BoolOps: target array should be scalar and have bool type!");

        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target._buffer, target._shapeInfo, target._bufferD, target._shapeInfoD);
    }

    void NDArray::reduceNumber(nd4j::reduce::LongOps op, NDArray& target, void *extraParams) const {

        if (isS())
            throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");
        if(!target.isScalar() || target._dataType != DataType::INT64)
            throw std::invalid_argument("NDArray::reduceNumber LongOps: target array should be scalar and have long type!");

        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }

//////////////////////////////////////////////////////////////////////////
    NDArray NDArray::indexReduceNumber(nd4j::indexreduce::Ops op, void *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::indexReduceNumber: you can't use this method on String array!");

        auto res = NDArrayFactory::create<Nd4jLong>(0);
        NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extraParams, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
        return res;
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
                        printf("%f,", arr->e<float>(row, col));
                    else if (arr->isZ())
                        printf("%lld,", arr->e<Nd4jLong>(row, col));
                    else if (arr->isB())
                        printf("%s,", arr->e<bool>(row, col)?"true":"false");
                    else if (arr->isS()) {
                        printf("\"%s\",", arr->e<std::string>(row * cols + col).c_str());
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

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        Nd4jLong* shapeInfo;
        if (_context->getWorkspace() == nullptr) {
            shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad.tadOnlyShapeInfo)];
        } else {
            shapeInfo = reinterpret_cast<Nd4jLong *>(_context->getWorkspace()->allocateBytes(shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)));
        }
        std::memcpy(shapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));

        auto array = new NDArray(bufferWithOffset(tad.tadOffsets[index]), shapeInfo, _context);
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

        ALLOCATE(newShapeInfo , _context->getWorkspace(), shapeInfoLength, Nd4jLong);
        memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

        auto newArr = new NDArray(_buffer, newShapeInfo, _context);
        newArr->_isShapeAlloc = true;
        newArr->_isBuffAlloc  = false;

        newArr->transposei();

        return newArr;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transp() const {
    int shapeInfoLength = shape::shapeInfoLength(rankOf());
    Nd4jLong* newShapeInfo;

    ALLOCATE(newShapeInfo , _context->getWorkspace(), shapeInfoLength, Nd4jLong);
    memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(Nd4jLong));

    NDArray newArr(_buffer, newShapeInfo, _context, false, true);

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
    target._isBuffAlloc = false;
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
    void NDArray::addRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::addRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType) && !(isR() && row->isR() && target->isR()))
            throw std::invalid_argument("NDArray::addRowVector: wrong type of target array !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
}

//////////////////////////////////////////////////////////////////////////
    void NDArray::subRowVector(const NDArray *row, NDArray * target) const {

        if (isS())
            throw std::runtime_error("NDArray::subRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::subRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::subRowVector: wrong type of target array !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Subtract, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
}

//////////////////////////////////////////////////////////////////////////
    void NDArray::mulRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::mulRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::mulRowVector: wrong type of target array !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
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

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Divide, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);

    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    void NDArray::addiRowVector(const NDArray *row) {

    if (isS())
        throw std::runtime_error("NDArray::addiRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
        throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::addColumnVector(const NDArray *column, NDArray *target) const {
        if (isS())
            throw std::runtime_error("NDArray::addColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, column->_dataType))
            throw std::invalid_argument("NDArray::addColumnVector: wrong type of target array !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    void NDArray::addiColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::addiColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    void NDArray::muliColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::muliColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, nullptr, nullptr);
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
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(dimensions.size()), Nd4jLong);

        char order = o == 'a' ? this->ordering() : o;

        if (order == 'c')
            shape::shapeBuffer(dimensions.size(), dataType(), dimensions.data(), newShape);
        else
            shape::shapeBufferFortran(dimensions.size(), dataType(), dimensions.data(), newShape);

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _context->getWorkspace());

        _shapeInfo = newShape;
        _isShapeAlloc = true;
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
        Nd4jLong* newShapeInfo = ShapeBuilders::copyShapeInfo(_shapeInfo, true, _context->getWorkspace());
        auto newArr = new NDArray(_buffer, newShapeInfo, _context, false, true);

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
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
        int8_t * newBuff = nullptr;
        ALLOCATE(newBuff, _context->getWorkspace(), shape::length(newShapeInfo) * sizeOfT(), int8_t);
        // assign new shape and new buffer to resulting array
        NDArray result(newBuff, newShapeInfo, _context, true, true);

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

//#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<resultLen; ++i) {

                auto xOffset = result.getOffset(i);
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
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
            delete []newShapeInfo;
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
        }
        RELEASE(newShapeInfo, _context->getWorkspace());

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const int ews = target.ews();
        const int targetLen = target.lengthOf();
        if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(Nd4jLong i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i*ews, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {

                auto xOffset = target.getOffset(i);
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
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
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for (int i = 0; i < targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else if(target.ordering() == 'c' && ews > 1) {
//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
                BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, templatedDoubleAssign, (target._buffer, i*ews, _buffer, yOffset), LIBND4J_TYPES, LIBND4J_TYPES);
            }
        }
        else {

//#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i=0;  i<targetLen; ++i) {

                auto xOffset = target.getOffset(i);
                auto yOffset = shape::subArrayIndex(target._shapeInfo, _shapeInfo, i);
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

        auto ret = new NDArray('c', outShape, _dataType,  _context);

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
            int retIdx = 0;
            if (isR()) {
                for (int k = 0; k < thisTensor.lengthOf(); k++) {
                    auto s = thisTensor.e<double>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor.p<double>(retIdx++, s);
                    }
                }
            } else {
                for (int k = 0; k < thisTensor.lengthOf(); k++) {
                    auto s = thisTensor.e<Nd4jLong>(k);
                    for (int j = 0; j < repeatDelta; j++) {
                        retTensor.p<Nd4jLong>(retIdx++, s);
                    }
                }
            }
        }
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
        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
        // create array to be returned
        auto ret = new NDArray(_buffer, shapeInfoNew, _context, false, true);
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
            RELEASE(target._buffer, target._context->getWorkspace());

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._context->getWorkspace());

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
            RELEASE(target._buffer, target._context->getWorkspace());

        auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target._context->getWorkspace());

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
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::e(i,j,k,l): one of input indexes is out of array length or rank!=4 !");

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
    NDArray scalar(_dataType, _context);
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
            BUILD_SINGLE_SELECTOR(xType, return NDArrayFactory::create_, (this->e<float>(pnt), this->_context), LIBND4J_TYPES);
        }

        if (idx.size() != this->rankOf())
            throw std::runtime_error("Number of indices should match");

        Nd4jLong* newShape;
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(this->rankOf()), Nd4jLong);
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

        auto result = new NDArray(bufferWithOffset(offset), newShape, this->_context);
        result->_isShapeAlloc = true;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::subarray(const std::initializer_list<NDIndex*>& idx) const {
        if (idx.size() != this->rankOf())
            throw std::runtime_error("NDArray::subarray: number of indices should match the array rank");

        Nd4jLong *newShape;
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(this->rankOf()), Nd4jLong);
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

        auto result = new NDArray(bufferWithOffset(offset), newShape, this->_context);
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
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(this->rankOf()), Nd4jLong);
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

        auto result = new NDArray(bufferWithOffset(offset), newShape, this->_context);
        result->_isShapeAlloc = true;

        return result;
    }


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


    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());
            
        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _context->getWorkspace());
        ArrayOptions::setDataType(newShape, DataTypeUtils::pickFloatingType(_dataType));
        auto result = new NDArray(newShape, true, _context, true);
        
        if(rankOf() == copy.size() || copy.empty())
            NativeOpExecutioner::execSummaryStatsScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), biasCorrected);
        else
            NativeOpExecutioner::execSummaryStats(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, copy.data(), copy.size(), nullptr, nullptr, biasCorrected);

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
            return varianceAlongDimension(op, biasCorrected, std::vector<int>(dimensions));
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
            NativeOpExecutioner::execSummaryStatsScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), biasCorrected);
        else
            NativeOpExecutioner::execSummaryStats(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, copy.data(), copy.size(), nullptr, nullptr, biasCorrected);
    }

    void NDArray::varianceAlongDimension(nd4j::variance::Ops op,const NDArray *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension(op, target, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    NDArray NDArray::operator()(const std::vector<Nd4jLong>& idx, bool keepUnitiesInShape)  const {

        const int rank = rankOf();
        Nd4jLong *newShape;
        ALLOCATE(newShape, _context->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
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

        NDArray result(bufferWithOffset(offset), newShape, _context, false, true);

        if(!keepUnitiesInShape) {

            std::vector<Nd4jLong> nonUnitDims;
            for (int d = 0; d < rank; ++d) {
                if(!(idx[2*d] != idx[2*d+1] && newShape[d+1] == 1))
                    nonUnitDims.push_back(newShape[d+1]);
            }
            if(nonUnitDims.size() != rank)
                result.reshapei(nonUnitDims);

            // std::vector<Nd4jLong> nonUnitDims = ShapeUtils<T>::evalDimsWithoutUnities(newShape);
            // if(nonUnitDims.size() != result.rankOf())
            //     result.reshapei(nonUnitDims);
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

        if (!Environment::getInstance()->isExperimentalBuild() && this->_dataType != other._dataType && (other._dataType != DataType::BOOL) ) {
            throw datatype_exception::build("NDArray::operator+: cannot add different types.", _dataType, other._dataType);
        }
        if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
            NDArray result(_shapeInfo, DataTypeUtils::pickPairwiseResultType(_shapeInfo, other._shapeInfo), false, this->_context);
            NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Add, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr);
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
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Subtract, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(_context, nd4j::scalar::Divide, _buffer, _shapeInfo, _bufferD, _shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

////////////////////////////////////////////////////////////////////////
ND4J_EXPORT NDArray operator-(const bfloat16& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<bfloat16>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const float& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const double& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const Nd4jLong& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<Nd4jLong>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator-(const int& scalar, const NDArray& arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator-: you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseSubtract, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const float& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<float>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const double& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<double>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    return result;
}

ND4J_EXPORT NDArray operator/(const int& scalar, const NDArray & arr) {
    if (arr.isS())
        throw std::runtime_error("NDArray::operator/: you can't use this method on String array!");
    if (arr.isB())
        throw std::runtime_error("NDArray::operator/: you can't divide scalar by bool array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.getShapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<int>()), false, arr.getContext());
    NativeOpExecutioner::execScalar(arr.getContext(), nd4j::scalar::ReverseDivide, arr.getBuffer(), arr.getShapeInfo(), arr.getSpecialBuffer(), arr.getSpecialShapeInfo(), result.getBuffer(), result.getShapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
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
}

void NDArray::operator*=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL)) {
        throw nd4j::datatype_exception::build("NDArray operator*=: Cannot multiply different types", this->dataType(), other.dataType());
    }

    if (!this->isScalar() && other.isScalar()) {
        NativeOpExecutioner::execScalar(_context, nd4j::scalar::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NativeOpExecutioner::execPairwiseTransform(_context, nd4j::pairwise::Multiply, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, nullptr);
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
        NDArray result(*ptr);
        delete ptr;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::assign(const NDArray& other, const Intervals& idx) {
        auto subarr = this->subarray(idx);
        subarr->assign(&other);
        delete subarr;
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

        NDArray result(const_cast<Nd4jLong*>(shapeInfo), false, _context, false);
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
        result->triggerAllocationFlag(true, true);

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

            auto array = new NDArray(bufferWithOffset(tad->tadOffsets[idx]), shapeInfo);
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
            auto array = new NDArray(bufferWithOffset(tad->tadOffsets[idx]), shapeInfo);
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

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::getOffset(const Nd4jLong i) const {

    if (i >= _length)
        throw std::invalid_argument("NDArray::getOffset: input index is out of array length !");

    auto ews   = this->ews();

    if(ordering() == 'c') {
        if(ews == 1)
            return i;
        if(ews > 1)
            return i * ews;
    }

    return shape::getIndexOffset(i, _shapeInfo, _length);
}

/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

