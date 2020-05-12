/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <loops/BroadcastPairwiseConverter.h>
#include <helpers/PointersManager.h>

namespace sd {

template <>
ND4J_EXPORT utf8string NDArray::e(const Nd4jLong i) const;
template <>
ND4J_EXPORT std::string NDArray::e(const Nd4jLong i) const;
template <>
ND4J_EXPORT std::u16string NDArray::e(const Nd4jLong i) const;
template <>
ND4J_EXPORT std::u32string NDArray::e(const Nd4jLong i) const;

////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {

    _context = other._context;
    _offset  = 0;

    setShapeInfo(ShapeDescriptor(other.dataType(), other.ordering(), other.shapeOf(), other.rankOf()));

    if(!isEmpty()) {
        _buffer = std::make_shared<DataBuffer>(other.lengthOf() * other.sizeOfT(), other.dataType(), other.getContext()->getWorkspace());
        this->assign(&other);
    }
    else
        _buffer = std::make_shared<DataBuffer>();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, sd::DataType dtype, sd::LaunchContext * context) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;
    _isAttached = _context->getWorkspace() != nullptr;
    _offset = 0;

    if (shape.empty())
        setShapeInfo(ShapeDescriptor::emptyDescriptor(dtype));
    else
        setShapeInfo(ShapeDescriptor(dtype, order, shape));

    _buffer = std::make_shared<DataBuffer>(lengthOf() * DataTypeUtils::sizeOf(dtype), dtype, getContext()->getWorkspace());
    _buffer->setToZeroBuffers();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, sd::DataType dtype, sd::LaunchContext * context) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;
    _offset  = 0;

    if (shape.size() == 0) {
        if (data.size() == 0)
            setShapeInfo(ShapeDescriptor::emptyDescriptor(dtype));
        else
            setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));
    } else {
        setShapeInfo(ShapeDescriptor(dtype, order, shape));
    }

    if (lengthOf() != data.size()) {
        nd4j_printf("NDArray constructor: data size [%i] doesn't match shape length [%i]\n", data.size(), lengthOf());
        throw std::runtime_error("Data size doesn't match shape");
    }

    _buffer = std::make_shared<DataBuffer>(lengthOf() * DataTypeUtils::sizeOf(dtype), dtype, getContext()->getWorkspace(), true);

    for(Nd4jLong i=0; i < lengthOf(); ++i) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedDoubleAssign<, double>(buffer(), i, reinterpret_cast<const void *>(data.data()), i), LIBND4J_TYPES);
    }
    tickWriteHost();
    syncToDevice();
}


////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, sd::LaunchContext* context) {

    _context = context;
    _offset  = 0;
    _isAttached = getContext()->getWorkspace() != nullptr;

    if (copyStrides)
        setShapeInfo(ShapeDescriptor(other->_shapeInfo));
    else
        setShapeInfo(ShapeDescriptor(other->dataType(), other->ordering(), other->shapeOf(), other->rankOf()));

    if (!isEmpty())
        _buffer = std::make_shared<DataBuffer>(lengthOf() * sizeOfT(), dataType(), getContext()->getWorkspace());
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape,  sd::DataType dtype, sd::LaunchContext * context, const bool isBuffAlloc) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;
    _offset  = 0;
    _isAttached = getContext()->getWorkspace() != nullptr;

    setShapeInfo(ShapeDescriptor(dtype, order, shape));

    _buffer = std::make_shared<DataBuffer>(buffer, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, getContext()->getWorkspace());
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(const Nd4jLong* shapeInfo, const sd::DataType dtype, const bool copyStrides, sd::LaunchContext * context, const bool nullify) {

    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;
    _offset  = 0;

    if (copyStrides)
        setShapeInfo(ShapeDescriptor(shapeInfo, dtype));
    else
        setShapeInfo(ShapeDescriptor(dtype, shape::order(shapeInfo), shape::shapeOf(shapeInfo), shape::rank(shapeInfo)));

    if (!isEmpty()) {
        _buffer = std::make_shared<DataBuffer>(lengthOf() * sizeOfT(), dtype, getContext()->getWorkspace());

        if (nullify)
            _buffer->setToZeroBuffers();
    }
}

////////////////////////////////////////////////////////////////////////
// scalar constructor
NDArray::NDArray(sd::DataType dtype, sd::LaunchContext* context, const bool isScalar) {

    _context = context;
    _offset  = 0;
    _isAttached = getContext()->getWorkspace() != nullptr;

    if (isScalar) {
        setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));
        _buffer = std::make_shared<DataBuffer>(sizeOfT(), dtype, getContext()->getWorkspace());
        _buffer->setToZeroBuffers();
    }
    else
        setShapeInfo(ConstantShapeHelper::getInstance()->emptyShapeInfo(dtype));
}

//////////////////////////////////////////////////////////////////////////
// move constructor
NDArray::NDArray(NDArray&& other) noexcept {

    _isView       = other._isView;
    _buffer       = other._buffer;
    _shapeInfo    = other._shapeInfo;
    _shapeInfoD   = other._shapeInfoD;
    _context      = other._context;
    _dataType     = other._dataType;
    _length       = other._length;
    _offset       = other._offset;

    other._buffer = std::make_shared<DataBuffer>();
    other._shapeInfo = other._shapeInfoD = nullptr;
    other._length = 0;
}

////////////////////////////////////////////////////////////////////////
//constructor, create empty array at given workspace
NDArray::NDArray(sd::LaunchContext * context) {
    _buffer    = std::make_shared<DataBuffer>();
    _shapeInfo = nullptr;
    _shapeInfoD = nullptr;
    _offset     = 0;
    _context = context;
    _length = 0;
}

    ////////////////////////////////////////////////////////////////////////
    // creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros, set dtype as array type
    NDArray::NDArray(const Nd4jLong* shapeInfo, const bool copyStrides, sd::LaunchContext * context, const bool nullify):
            NDArray(shapeInfo, ArrayOptions::dataType(shapeInfo), copyStrides, context) {
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray::NDArray(std::shared_ptr<DataBuffer> buffer, const ShapeDescriptor& descriptor, sd::LaunchContext* context, const Nd4jLong offset) {

        _context = context;
        _offset  = offset;

        setShapeInfo(descriptor);

        _buffer = buffer;

        _isView = offset > 0 || _length * DataTypeUtils::sizeOf(_dataType) < buffer->getLenInBytes();
    }

    NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, sd::LaunchContext * context, const bool isBuffAlloc) : NDArray::NDArray(buffer, const_cast<const Nd4jLong *>(shapeInfo), context, isBuffAlloc) {
        //
    }

    ////////////////////////////////////////////////////////////////////////
    // do not allocate memory, memory for array is passed from outside
    NDArray::NDArray(void *buffer, const Nd4jLong *shapeInfo, sd::LaunchContext * context, const bool isBuffAlloc) {

        if (buffer == nullptr && ArrayOptions::arrayType(shapeInfo) != ArrayType::EMPTY)
            throw std::runtime_error("NDArray constructor: can't be initalized with nullptr buffer !");

        if (shapeInfo == nullptr)
            throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo !");

        if ((int) shapeInfo[0] > MAX_RANK)
            throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");

        _context = context;
        _isAttached = getContext()->getWorkspace() != nullptr;
        _offset = 0;

        setShapeInfo(ShapeDescriptor(shapeInfo));

        if (this->isEmpty()) {
            tickReadDevice();
            tickReadHost();
        }
        else {
            _buffer = std::make_shared<DataBuffer>(buffer, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, getContext()->getWorkspace());
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // do not allocate memory, memory for array is passed from outside
    // we suppose the content of both (device and host) buffers is identical
    NDArray::NDArray(void *buffer, void* bufferD, const Nd4jLong *shapeInfo, sd::LaunchContext * context, const bool isBuffAlloc, const bool isBuffDAlloc) {

        if (shapeInfo == nullptr)
            throw std::runtime_error("NDArray constructor cuda: can't be initalized without shapeinfo");

        if ((int) shapeInfo[0] > MAX_RANK)
            throw std::invalid_argument("NDArray constructor cuda: rank of NDArray can't exceed 32");

        _context = context;
        _offset  = 0;

        setShapeInfo(ShapeDescriptor(shapeInfo));

        if (!isEmpty())
            _buffer = std::make_shared<DataBuffer>(buffer, bufferD, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, isBuffDAlloc, getContext()->getWorkspace());
    }

//////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::shared_ptr<DataBuffer> buffer, const char order, const std::vector<Nd4jLong> &shape, sd::LaunchContext* context) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32");

    _context = context;
    _offset  = 0;

    setShapeInfo(ShapeDescriptor(buffer->getDataType(), order, shape));

    _buffer = buffer;

    _isView =  _length * DataTypeUtils::sizeOf(_dataType) < buffer->getLenInBytes();
}
/////////////////////////////////////////////////////////////////////////
// u16 string constructors
NDArray::NDArray(const std::u16string& u16string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype)) {
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
    }

    if (!unicode::isStringValidU16(u16string.data(), u16string.data() + u16string.size())) {
        throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
    }

    // one word that is why used 1
    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

    Nd4jLong dataLength = [&] {
        if (dtype == DataType::UTF16) {
            return static_cast<Nd4jLong>(u16string.size() * sizeof(uint16_t));
        }
        if (dtype == DataType::UTF32) {
            return unicode::offsetUtf16StringInUtf32(u16string.data(), u16string.size());
        }
        return unicode::offsetUtf16StringInUtf8(u16string.data(), u16string.size());
    }();

    Nd4jLong offsets[2] = { 0 , dataLength };

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _isAttached = getContext()->getWorkspace() != nullptr;
    _offset = 0;

    setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));

    memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);
    if (dtype == DataType::UTF8) {
        unicode::utf16to8(u16string.data(), data, u16string.size());
    }
    else if (dtype == DataType::UTF16) {
        memcpy(data, u16string.data(), dataLength);
    }
    else {
        unicode::utf16to32(u16string.data(), data, u16string.size());
    }

    tickWriteHost();
    syncToDevice();
}

/////////////////////////////////////////////////////////////////////////
// u32 string constructors
NDArray::NDArray(const std::u32string& u32string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype)) {
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
    }

    if (!unicode::isStringValidU32(u32string.data(), u32string.data() + u32string.size())) {
        throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
    }
    // one word that is why used 1
    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

    Nd4jLong dataLength = [&] {
        if (dtype == DataType::UTF16) {
            return unicode::offsetUtf32StringInUtf16(u32string.data(), u32string.size());
        }
        if (dtype == DataType::UTF32) {
            return static_cast<Nd4jLong>(sizeof(uint32_t) * u32string.size());
        }
        return unicode::offsetUtf32StringInUtf8(u32string.data(), u32string.size());
    }();

    Nd4jLong offsets[2] = { 0 , dataLength };

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _isAttached = getContext()->getWorkspace() != nullptr;
    _offset = 0;

    setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));

    memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);
    if (dtype == DataType::UTF8) {
        unicode::utf32to8(u32string.data(), data, u32string.size());
    }
    else if (dtype == DataType::UTF16) {
        unicode::utf32to16(u32string.data(), data, u32string.size());
    }
    else {
        memcpy(data, u32string.data(), u32string.size() * sizeof(uint32_t));
    }

    tickWriteHost();
    syncToDevice();
}

/////////////////////////////////////////////////////////////////////////
// u8 string constructors
NDArray::NDArray(const std::string& str, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype)) {
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
    }

    if (!unicode::isStringValidU8(str.data(), str.data() + str.size())) {
        throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
    }

    // one word that is why used 1
    auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

    Nd4jLong dataLength = [&] {
        if (dtype == DataType::UTF16) {
            return unicode::offsetUtf8StringInUtf16(str.data(), str.size());
        }
        if (dtype == DataType::UTF32) {
            return unicode::offsetUtf8StringInUtf32(str.data(), str.size());
        }
        return static_cast<Nd4jLong>(str.size());
    }();

    Nd4jLong offsets[2] = { 0 , dataLength };

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _isAttached = getContext()->getWorkspace() != nullptr;
    _offset = 0;

    setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));

    memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    if (dtype == DataType::UTF8) {
        memcpy(data, str.data(), str.size());
    }
    else if (dtype == DataType::UTF16) {
        unicode::utf8to16(str.data(), data, str.size());
    }
    else {
        unicode::utf8to32(str.data(), data, str.size());
    }

    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
// constructors for vector of  strings
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<const char*>& string, const sd::DataType dataType, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dataType))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (const auto& str : string) {
        if (!unicode::isStringValidU8(str, str + std::char_traits<char>::length(str)) ) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);
    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dataType == DataType::UTF16)
                return unicode::offsetUtf8StringInUtf16(string[e], std::char_traits<char>::length(string[e]));
            if (dataType == DataType::UTF32)
                return  unicode::offsetUtf8StringInUtf32(string[e], std::char_traits<char>::length(string[e]));
            return static_cast<Nd4jLong>(std::char_traits<char>::length(string[e]));
        }();
    }
    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dataType, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dataType, 'c', shape));

    _isView = false;

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
                auto cdata = data + offsets[e];
                if (dataType == DataType::UTF16) {
                    unicode::utf8to16(string[e], cdata, std::char_traits<char>::length(string[e]));
                }
                else if (dataType == DataType::UTF32) {
                    unicode::utf8to32(string[e], cdata, std::char_traits<char>::length(string[e]));
                }
                else {
                    memcpy(cdata, string[e], std::char_traits<char>::length(string[e]));
                }
        }
    };

    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<std::string>& string, const sd::DataType dataType, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dataType))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (const auto& str : string) {
        if (!unicode::isStringValidU8(str.data(), str.data() + str.size())) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);
    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dataType == DataType::UTF16)
                return unicode::offsetUtf8StringInUtf16(string[e].data(), string[e].size());
            if (dataType == DataType::UTF32)
                return  unicode::offsetUtf8StringInUtf32(string[e].data(), string[e].size());
            return static_cast<Nd4jLong>(string[e].size());
        }();
    }

    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dataType, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dataType, 'c', shape));

    _isView = false;

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
             auto cdata = data + offsets[e];
             if (dataType == DataType::UTF16) {
                 unicode::utf8to16(string[e].data(), cdata, string[e].size());
             }
             else if (dataType == DataType::UTF32) {
                 unicode::utf8to32(string[e].data(), cdata, string[e].size());
             }
             else {
                 memcpy(cdata, string[e].data(), string[e].size());
             }
        }
    };

    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);


    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<std::u16string>& string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (const auto& str : string) {
        if (!unicode::isStringValidU16(str.data(), str.data() + str.size())) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);
    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dtype == DataType::UTF16)
                return static_cast<Nd4jLong>(sizeof(uint16_t) * string[e].size());
            if (dtype == DataType::UTF32)
                return unicode::offsetUtf16StringInUtf32(string[e].data(), string[e].size());
            return unicode::offsetUtf16StringInUtf8(string[e].data(), string[e].size());
        }();
    }
    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dtype, 'c', shape));

    _isView = false;

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
             auto cdata = data + offsets[e];
             if (dtype == DataType::UTF16) {
                 memcpy(cdata, string[e].data(), string[e].size() * sizeof(uint16_t));
             }
             else if (dtype == DataType::UTF32) {
                 unicode::utf16to32(string[e].data(), cdata, string[e].size());
             }
             else {
                 unicode::utf16to8(string[e].data(), cdata, string[e].size());
             }
        }
    };
    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<const char16_t*>& string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (const auto& str : string) {
        if (!unicode::isStringValidU16(str, str + std::char_traits<char16_t>::length(str))) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);
    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dtype == DataType::UTF16)
                return static_cast<Nd4jLong>(sizeof(uint16_t) * std::char_traits<char16_t>::length(string[e]));
            if (dtype == DataType::UTF32)
                return unicode::offsetUtf16StringInUtf32(string[e], std::char_traits<char16_t>::length(string[e]));
            return unicode::offsetUtf16StringInUtf8(string[e], std::char_traits<char16_t>::length(string[e]));
        }();
    }
    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dtype, 'c', shape));

    _isView = false;

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);


    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
             auto cdata = data + offsets[e];
             if (dtype == DataType::UTF16) {
                 memcpy(cdata, string[e], std::char_traits<char16_t>::length(string[e]) * sizeof(uint16_t));
             }
             else if (dtype == DataType::UTF32) {
                 unicode::utf16to32(string[e], cdata, std::char_traits<char16_t>::length(string[e]));
             }
             else {
                 unicode::utf16to8(string[e], cdata, std::char_traits<char16_t>::length(string[e]));
             }
        }
    };
    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<std::u32string>& string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (auto str : string) {
        if (!unicode::isStringValidU32(str.data(), str.data() + str.size())) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);

    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dtype == DataType::UTF16)
                return unicode::offsetUtf32StringInUtf16(string[e].data(), string[e].size());
            if (dtype == DataType::UTF32)
                return static_cast<Nd4jLong>(sizeof(uint32_t) * string[e].size());
            return unicode::offsetUtf32StringInUtf16(string[e].data(), string[e].size());
        }();
    }
    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dtype, 'c', shape));

    _isView = false;

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
            auto cdata = data + offsets[e];
            if (dtype == DataType::UTF16) {
                unicode::utf32to16(string[e].data(), cdata, string[e].size());
            }
            else if (dtype == DataType::UTF32) {
                memcpy(cdata, string[e].data(), string[e].size() * sizeof(uint32_t));
            }
            else {
                unicode::utf32to8(string[e].data(), cdata, string[e].size());
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    tickWriteHost();
    syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const std::vector<Nd4jLong>& shape, const std::vector<const char32_t *>& string, sd::DataType dtype, sd::LaunchContext* context) {

    if (!DataTypeUtils::isS(dtype))
        throw std::invalid_argument("NDArray::NDArray: invalid DataType used");

    if (shape::prodLong(shape.data(), shape.size()) != string.size())
        throw std::invalid_argument("NDArray::NDArray: Number of strings should match length of array");

    for (const auto& str : string) {
        if (!unicode::isStringValidU32(str, str + std::char_traits<char32_t>::length(str))) {
            throw std::invalid_argument("NDArray::NDArray: invalid character in input string");
        }
    }

    Nd4jLong headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

    std::vector<Nd4jLong> offsets(string.size() + 1);

    Nd4jLong dataLength = 0;
    for (int e = 0; e < string.size(); e++) {
        offsets[e] = dataLength;
        dataLength += [&] {
            if (dtype == DataType::UTF16)
                return unicode::offsetUtf32StringInUtf16(string[e], std::char_traits<char32_t>::length(string[e]));
            if (dtype == DataType::UTF32)
                return static_cast<Nd4jLong>(sizeof(uint32_t) * std::char_traits<char32_t>::length(string[e]));
            return unicode::offsetUtf32StringInUtf16(string[e], std::char_traits<char32_t>::length(string[e]));
        }();
    }
    offsets[string.size()] = dataLength;

    _buffer = std::make_shared<DataBuffer>(headerLength + dataLength, dtype, context->getWorkspace(), true);

    _context = context;
    _offset = 0;

    setShapeInfo(ShapeDescriptor(dtype, 'c', shape));

    _isView = _length * DataTypeUtils::sizeOf(_dataType) < _buffer->getLenInBytes();

    setAttached(context->getWorkspace() != nullptr);

    memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength);

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {
            auto cdata = data + offsets[e];
            if (dtype == DataType::UTF16) {
                unicode::utf32to16(string[e], cdata, std::char_traits<char32_t>::length(string[e]));
            }
            else if (dtype == DataType::UTF32) {
                memcpy(cdata, string[e], std::char_traits<char32_t>::length(string[e]) * sizeof(uint32_t));
            }
            else {
                unicode::utf32to8(string[e], cdata, std::char_traits<char32_t>::length(string[e]));
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    tickWriteHost();
    syncToDevice();
}

////////////////////////////////////////////////////////////////////////
// assignment operator
    NDArray& NDArray::operator=(const NDArray& other) {

    if (this == &other || (_shapeInfo == other._shapeInfo && _shapeInfo == nullptr))
        return *this;

    if (_shapeInfo != nullptr && shape::equalsTypesAndShapesSoft(_shapeInfo, other._shapeInfo)) {
        if(!other.isEmpty())
            this->assign(&other);
    }
    else {
        _context = other._context;
        _offset  = 0;
        setShapeInfo(ShapeDescriptor(other.dataType(), other.ordering(), other.shapeOf(), other.rankOf()));

        if(!other.isEmpty()) {
            _buffer = std::make_shared<DataBuffer>(other.lengthOf() * other.sizeOfT(), other.dataType(), other.getContext()->getWorkspace());
            this->assign(&other);
        }
        else
            _buffer = std::make_shared<DataBuffer>();
    }
    return *this;
}


//////////////////////////////////////////////////////////////////////////
bool NDArray::isC() const {
    // TODO: this method must be implemented once we add support for complex numbers
    return false;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isS() const {
    return (dataType() == DataType::UTF8 ||
            dataType() == DataType::UTF16 ||
            dataType() == DataType::UTF32);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isR() const {
    auto xType = ArrayOptions::dataType(this->_shapeInfo);
    return xType == FLOAT32 || xType == HALF || xType == DOUBLE || xType == FLOAT8 || xType == BFLOAT16;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isZ() const {
    // TODO: decide if we really want to exclude Bool here
    return !isC() && !isR() && !isB() && !isS();
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isB() const {
    return ArrayOptions::dataType(this->_shapeInfo) == BOOL;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
std::string NDArray::toStringValue(T value) {
    std::ostringstream os ;
    //throw the value into the string stream
    os << value ;
    //convert the string stream into a string and return
    return os.str() ;
}

//////////////////////////////////////////////////////////////////////////
template<>
std::string NDArray::toStringValue(float16 value) {
    std::ostringstream os ;
    //throw the value into the string stream
    os << (float) value ;
    //convert the string stream into a string and return
    return os.str() ;
}

//////////////////////////////////////////////////////////////////////////
template<>
std::string NDArray::toStringValue(bfloat16 value) {
    std::ostringstream os ;
    //throw the value into the string stream
    os << (float) value ;
    //convert the string stream into a string and return
    return os.str() ;
}

//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
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
        else if (this->isS()) // todo add utf16 and utf32
            os << this->e<std::string>(e);
        if (e < limit - 1)
            os << ", ";
    }
    os << "]";
    return os.str();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::vector<T> NDArray::getBufferAsVector() const {
    std::vector<T> vector(lengthOf());
    for (Nd4jLong e = 0; e < lengthOf(); e++)
        vector[e] = this->e<T>(e);
    return vector;
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT std::vector, NDArray::getBufferAsVector() const, LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeAsFlatVector() const {
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
std::vector<int> NDArray::getShapeAsVectorInt() const {

    std::vector<int> vector(this->rankOf());
    for (int e = 0; e < this->rankOf(); e++)
        vector[e] = static_cast<int>(this->sizeAt(e));

    return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeInfoAsFlatVector() const {
    int magicNumber = shape::shapeInfoLength(this->rankOf());
    std::vector<int64_t> vector(magicNumber);

    for (int e = 0; e < magicNumber; e++)
        vector[e] = static_cast<int64_t>(_shapeInfo[e]);

    return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<Nd4jLong> NDArray::getShapeInfoAsVector() const {
    int magicNumber = shape::shapeInfoLength(this->rankOf());
    std::vector<Nd4jLong> vector(magicNumber);
    for (int e = 0; e < magicNumber; e++)
        vector[e] = this->_shapeInfo[e];
    return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<int8_t> NDArray::asByteVector() {

    if (isS()) {
        // string data type requires special treatment
        syncToHost();
        auto numWords = this->lengthOf();
        auto offsetsBuffer = this->bufferAsT<Nd4jLong>();
        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(numWords);
        auto dataLength = offsetsBuffer[numWords];
        std::vector<int8_t> result(headerLength + dataLength);

        memcpy(result.data(), buffer(), headerLength + dataLength);

        return result;
    } else {
        // all other types are linear
        std::vector<int8_t> result((unsigned long long) this->lengthOf() * sizeOfT());

        if (this->isView()) {
            auto tmp = this->dup(this->ordering());
            syncToHost();
            memcpy(result.data(), tmp.buffer(), (unsigned long long) lengthOf() * sizeOfT());
        } else {
            syncToHost();
            memcpy(result.data(), buffer(), (unsigned long long) lengthOf() * sizeOfT());
        }
        return result;
    }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::linspace(const double start) {
    linspace(start, 1);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::linspace(const double start, const double step) {
    if (isS())
        throw std::runtime_error("NDArray::linspace: you can't use this method on String array!");
    Nd4jLong numElements = this->lengthOf();
    for (Nd4jLong e = 0; e < numElements; e++)
        this->p(e, start + (step * e));
}

////////////////////////////////////////////////////////////////////////
void NDArray::streamline(char o) {
    char order = o == 'a' ? this->ordering() : o;
    syncToDevice();
    std::shared_ptr<DataBuffer> newBuffer = std::make_shared<DataBuffer>(this->lengthOf() * sizeOfT(), dataType(), getContext()->getWorkspace());
    auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(dataType(), order, rankOf(), shapeOf());
    NativeOpExecutioner::execTransformSame(getContext(), transform::Copy, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), newBuffer->primary(), static_cast<Nd4jLong*>(shapeBuffer.primary()), newBuffer->special(), static_cast<Nd4jLong*>(shapeBuffer.special()), nullptr, nullptr, nullptr);
    setShapeInfo(static_cast<Nd4jLong*>(shapeBuffer.primary()));
    _buffer = newBuffer;
    _offset = 0;
    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
NDArray& NDArray::operator=(NDArray&& other) noexcept {
    if (this == &other)
        return *this;

    _isView       = other._isView;
    _buffer       = other._buffer;
    _shapeInfo    = other._shapeInfo;
    _shapeInfoD   = other._shapeInfoD;
    _context      = other._context;
    _dataType     = other._dataType;
    _length       = other._length;
    _offset       = other._offset;

    other._buffer = std::make_shared<DataBuffer>();
    other._shapeInfo = other._shapeInfoD = nullptr;
    other._length = 0;

    return *this;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray& NDArray::operator=(const T scalar) {
    this->assign(scalar);
    return *this;
}
template ND4J_EXPORT NDArray& NDArray::operator=(const double scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const float scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const float16 scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const bfloat16 scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const Nd4jLong scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const int scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const int8_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const uint8_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const uint16_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const uint32_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const uint64_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const int16_t scalar);
template ND4J_EXPORT NDArray& NDArray::operator=(const bool scalar);

//////////////////////////////////////////////////////////////////////////
void NDArray::copyBuffersContinuouslyFrom(const NDArray& other, size_t sizeToCopyInBytes, Nd4jLong offsetThis, Nd4jLong offsetOther) {

    if(offsetThis == 0)
        offsetThis = bufferOffset();
    if(offsetOther == 0)
        offsetOther = other.bufferOffset();

    dataBuffer()->copyBufferFrom(*other.getDataBuffer(), sizeToCopyInBytes, offsetThis, offsetOther);
}

////////////////////////////////////////////////////////////////////
// This method assigns values of given NDArray to this one
void NDArray::assign(const NDArray& other, bool allowParallelism) {

    if (this == &other)
        return;

    if (other.isEmpty()) {
        if (!isEmpty()) {
             throw std::runtime_error("Cannot assign empty array to non-empty array");
        }
        return;
    }

    if(isEmpty()) {
        *this = other;
        return;
    }

    if (other.lengthOf() == 1) {

        if(lengthOf() == 1) {
            NDArray::preparePrimaryUse({this}, {&other});
            BUILD_DOUBLE_SELECTOR(dataType(), other.dataType(), templatedDoubleAssign, (buffer(), 0, other.buffer(), 0), LIBND4J_TYPES, LIBND4J_TYPES);
            NDArray::registerPrimaryUse({this}, {&other});
            this->syncToDevice();
        }
        else {
            if (dataType() != other.dataType()) {
                auto tmp = other.cast(dataType());
                NDArray::prepareSpecialUse({this}, {&tmp});
                NativeOpExecutioner::execScalar(getContext(), scalar::CopyPws, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr, allowParallelism);
                NDArray::registerSpecialUse({this}, {});
            }
            else {
                NDArray::prepareSpecialUse({this}, {&other});
                NativeOpExecutioner::execScalar(getContext(), scalar::CopyPws, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr, allowParallelism);
                NDArray::registerSpecialUse({this}, {&other});
            }
        }
    }
    else {
        if (other.lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("NDArray::assign: lengths of arrays are mismatched");
        }

        NDArray::prepareSpecialUse({this}, {&other});
        NativeOpExecutioner::execTransformAny(getContext(), transform::Assign, other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, nullptr, nullptr, allowParallelism);
        NDArray::registerSpecialUse({this}, {&other});
    }
}

//////////////////////////////////////////////////////////////////////////
// This method assigns values of given NDArray to this one, wrt order
void NDArray::assign(const NDArray *other, bool allowParallelism) {
    assign(*other, allowParallelism);
}

//////////////////////////////////////////////////////////////////////////
template <typename T, typename>
void NDArray::assign(const T& value, bool allowParallelism) {
    // just fire scalar
    auto temp = NDArrayFactory::create(dataType(), value, this->getContext());

    NDArray::prepareSpecialUse({this}, {&temp});
    NativeOpExecutioner::execScalar(getContext(), sd::scalar::CopyPws, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), temp.buffer(), temp.shapeInfo(), temp.specialBuffer(), temp.specialShapeInfo(), nullptr, allowParallelism);
    NDArray::registerSpecialUse({this}, {&temp});
}
template ND4J_EXPORT void NDArray::assign(const double& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const float& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const float16& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const bfloat16& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const Nd4jLong& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const int& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const int8_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const int16_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const uint8_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const uint16_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const uint32_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const uint64_t& value, bool allowParallelism);
template ND4J_EXPORT void NDArray::assign(const bool& value, bool allowParallelism);

//////////////////////////////////////////////////////////////////////////
NDArray* NDArray::detach() {

    if (!isAttached())
        return this;

    std::shared_ptr<DataBuffer> newBuffer = std::make_shared<DataBuffer>(lengthOf() * sizeOfT(), dataType());

    auto result = new NDArray(newBuffer, ShapeDescriptor(dataType(), ordering(), shapeOf(), rankOf()));

    result->assign(*this);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceNumber(sd::variance::Ops op, bool biasCorrected) {

    NDArray res(DataTypeUtils::pickFloatingType(dataType()), getContext());

    NDArray::prepareSpecialUse({&res}, {this});
    NativeOpExecutioner::execSummaryStatsScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo(), biasCorrected);
    NDArray::registerSpecialUse({&res}, {this});

    return res;
}

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
NDArray NDArray::sumNumber() const {
    if (isS())
        throw std::runtime_error("NDArray::sumNumber: you can't use this method on String array!");
    NDArray res(dataType(), getContext());

    NDArray::prepareSpecialUse({&res}, {this});
    NativeOpExecutioner::execReduceSameScalar(getContext(), sd::reduce::SameOps::Sum, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
    NDArray::registerSpecialUse({&res}, {this});

    return res;
}

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
NDArray NDArray::meanNumber() const {

    if (isS())
        throw std::runtime_error("NDArray::meanNumber: you can't use this method on String array!");
    NDArray res(DataTypeUtils::pickFloatingType(dataType()), getContext());

    NDArray::prepareSpecialUse({&res}, {this});
    NativeOpExecutioner::execReduceFloatScalar(getContext(), sd::reduce::FloatOps::Mean, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
    NDArray::registerSpecialUse({&res}, {this});
    return res;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::hasNaNs() {
    if (isS())
        throw std::runtime_error("NDArray::hasNaNs: you can't use this method on String array!");
    return this->reduceNumber(sd::reduce::IsNan, nullptr).e<int>(0) > 0;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::hasInfs() {
    if (isS())
        throw std::runtime_error("NDArray::hasInfs: you can't use this method on String array!");
    return this->reduceNumber(sd::reduce::IsInf, nullptr).e<int>(0) > 0;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isFinite() {
    if (isS())
        throw std::runtime_error("NDArray::isFinite: you can't use this method on String array!");
    return this->reduceNumber(sd::reduce::IsInfOrNan, nullptr).e<int>(0) == 0;
}

//////////////////////////////////////////////////////////////////////////
template <typename T, typename Y>
void NDArray::templatedSet(void *buffer, const Nd4jLong *indices, const void *value) {
    auto t = reinterpret_cast<T *>(buffer);
    const auto y = *(reinterpret_cast<const Y *>(value));

    auto xOffset = shape::getOffset(shapeInfo(), indices);
    t[xOffset] = static_cast<T>(y);
}
BUILD_DOUBLE_TEMPLATE(template ND4J_EXPORT void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, const void *value), LIBND4J_TYPES, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T, typename Y>
void NDArray::templatedSet(void *buffer, const Nd4jLong offset, const void *value) {
    auto t = reinterpret_cast<T *>(buffer);
    const auto y = *(reinterpret_cast<const Y *>(value));

    t[offset] = static_cast<T>(y);
}
BUILD_DOUBLE_TEMPLATE(template ND4J_EXPORT void NDArray::templatedSet, (void *buffer, const Nd4jLong offset, const void *value), LIBND4J_TYPES, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void NDArray::setContext(sd::LaunchContext  *context) {

    _context = context;
    if (getContext() == nullptr)
        _context = sd::LaunchContext ::defaultContext(); // empty context for default cases
}

//////////////////////////////////////////////////////////////////////////
void const* NDArray::bufferWithOffset(Nd4jLong offset) const {
    return const_cast<int8_t *>(buffer() != nullptr ? static_cast<const int8_t*>(buffer()) + (offset * sizeOfT()) : nullptr);
}

//////////////////////////////////////////////////////////////////////////
void* NDArray::bufferWithOffset(Nd4jLong offset) {
    return const_cast<int8_t *>(buffer() != nullptr ? static_cast<const int8_t*>(buffer()) + (offset * sizeOfT()) : nullptr);
}

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
NDArray NDArray::reduceAlongDimension(sd::reduce::FloatOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, isR() ? dataType() : Environment::getInstance()->defaultFloatDataType(), keepDims, supportOldShapes, getContext()->getWorkspace());

    NDArray result(newShape, true, getContext());

    this->reduceAlongDimension(op, result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, getContext()->getWorkspace());

    NDArray result(newShape, true, getContext());

    reduceAlongDimension(op, result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::BOOL, keepDims, supportOldShapes, getContext()->getWorkspace());

    NDArray result(newShape, true, getContext());

    reduceAlongDimension(op, result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {

    std::vector<int> copy(dimensions);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::INT64, keepDims, supportOldShapes, getContext()->getWorkspace());

    NDArray result(newShape, true, getContext());

    reduceAlongDimension(op, result, copy, keepDims, supportOldShapes, false);

    return result;
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
NDArray NDArray::reduceAlongDimension(sd::reduce::FloatOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::SameOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::BoolOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::LongOps op, const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
    return reduceAlongDimension(op, std::vector<int>(dimensions), keepDims, supportOldShapes);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::FloatOps op, void *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");

    auto shape = ConstantShapeHelper::getInstance()->scalarShapeInfo(DataTypeUtils::pickFloatingType(dataType()));
    NDArray result(shape, true, this->getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::SameOps op, void *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");

    NDArray result(dataType(), getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::BoolOps op, void *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");

    auto shape = ConstantShapeHelper::getInstance()->scalarShapeInfo(DataType::BOOL);
    NDArray result(shape, true, this->getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::LongOps op, void *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");

    auto shape = ConstantShapeHelper::getInstance()->scalarShapeInfo(DataType::INT64);
    NDArray result(shape, true, this->getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::FloatOps op, NDArray& target, void *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::reduceNumber FloatOps: you can't use this method on String array!");
    if(target.lengthOf() != 1 || target.dataType() != DataTypeUtils::pickFloatingType(dataType()))
        throw std::invalid_argument("NDArray::reduceNumber FloatOps: target array should be scalar and have corresponding float type!");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::SameOps op, NDArray& target, void *extraParams) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceNumber SameOps: you can't use this method on String array!");
    if(target.lengthOf() != 1 || target.dataType() != dataType())
        throw std::invalid_argument("NDArray::reduceNumber SameOps: target array should be scalar and have same type as this array!");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::BoolOps op, NDArray& target, void *extraParams) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceNumber BoolOps: you can't use this method on String array!");
    if(target.lengthOf() != 1 || target.dataType() != DataType::BOOL)
        throw std::invalid_argument("NDArray::reduceNumber BoolOps: target array should be scalar and have bool type!");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::LongOps op, NDArray& target, void *extraParams) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceNumber LongOps: you can't use this method on String array!");
    if(target.lengthOf() != 1 || target.dataType() != DataType::INT64)
        throw std::invalid_argument("NDArray::reduceNumber LongOps: target array should be scalar and have long type!");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::indexReduceNumber(sd::indexreduce::Ops op, ExtraArguments *extraParams) {
    if (isS())
        throw std::runtime_error("NDArray::indexReduceNumber: you can't use this method on String array!");

    auto res = NDArrayFactory::create<Nd4jLong>(0);

    NDArray::NDArray::prepareSpecialUse({&res}, {this});
    NativeOpExecutioner::execIndexReduceScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams == nullptr ? nullptr : extraParams->argumentsAsT(this->dataType()), res.buffer(), res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
    NDArray::NDArray::registerSpecialUse({&res}, {this});

    return res;
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::tensorsAlongDimension(std::initializer_list<int> dimensions) const {
    return tensorsAlongDimension(std::vector<int>(dimensions));
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::tensorsAlongDimension(const std::vector<int>& dimensions) const {
    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
    Nd4jLong numTads = this->lengthOf() / tadLength;

    return numTads;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::printShapeInfo(const char * msg) const {

    int rank = shape::rank(_shapeInfo);
    int lim = shape::shapeInfoLength(rank);

    if(msg != nullptr)
        printf("shapeInfo %s: [", msg);
    else
        printf("shapeInfo: [");

    printf("%i,  ", rank);
    for (int i = 1; i < shape::shapeInfoLength(rank) - 3; i++){
        if(i == rank + 1)
            printf("  ");
        printf("%lld,", _shapeInfo[i]);
    }
    printf("  %lld,", shape::type(_shapeInfo));
    printf("%lld,", shape::elementWiseStride(_shapeInfo));
    printf("%lld]\n", (Nd4jLong)shape::order(_shapeInfo));

    fflush(stdout);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::printBuffer(const char* msg, Nd4jLong limit, const bool sync) const{
    if (sync)
        syncToHost();

    if (limit == -1)
        limit = (int) this->lengthOf();

    if (msg != nullptr)
        printf("%s: [", msg);
    else
        printf("[");
    if (this->isR()) {
        for (Nd4jLong e = 0; e < limit; e++) {
            if (e)
                printf(", ");
            printf("%f", this->e<float>(e));
        }
    }
    else if (this->isZ()) {
        for (Nd4jLong e = 0; e < limit; e++) {
            if (this->dataType() != sd::DataType::INT64 && this->dataType() != sd::DataType::UINT64)
                printf("%d", this->e<int>(e));
            else
                printf("%llu", this->e<Nd4jLong>(e));
            if (e < limit - 1)
                printf(", ");
        }
    }
    else if (this->isB()) {
        for (Nd4jLong e = 0; e < limit; e++) {
            if (this->e<bool>(e))
                printf("true");
            else
                printf("false");
            if (e < limit - 1)
                printf(", ");
        }
    }
    else if (this->isS()) {
        // todo do we need this print offsets
        /*
        for (Nd4jLong e = 0; e < limit; e++) {
            printf("\"%lld\"", this->getOffset(e));
            if (e < limit - 1)
                printf(", ");
        }
        printf("]\n[");
        */
        for (Nd4jLong e = 0; e < limit; e++) {
            printf("\"%s\"", this->e<std::string>(e).c_str());
            if (e < limit - 1)
                printf(", ");
        }
    }
    printf("]\n");
    fflush(stdout);
}

//////////////////////////////////////////////////////////////////////////
// print element by element consequently in a way they (elements) are stored in physical memory
void NDArray::printLinearBuffer() const {

    syncToHost();

    const auto ews = this->ews() > 0 ? this->ews() : 1;
    const auto len = this->lengthOf();

    printf("[");

    if (this->dataType() == sd::DataType::INT32) {
        for(Nd4jLong e = 0; e < len; e++)
            printf("%d, ", this->bufferAsT<int>()[e * ews]);
    }
    else if(this->dataType() == sd::DataType::INT64) {
        for(Nd4jLong e = 0; e < len; e++)
            printf("%lld, ", this->bufferAsT<Nd4jLong>()[e * ews]);
    }
    else if(this->dataType() == sd::DataType::FLOAT32) {
        for(Nd4jLong e = 0; e < len; e++)
            printf("%.3f, ", this->bufferAsT<float>()[e * ews]);
    }
    else if(this->dataType() == sd::DataType::DOUBLE) {
        for(Nd4jLong e = 0; e < len; e++)
            printf("%.3f, ", this->bufferAsT<double>()[e * ews]);
    }
    else
        throw std::invalid_argument("NDArray::printLinearBuffer: not implemented yet for this data type !");

    printf("]\n");
    fflush(stdout);
}
//////////////////////////////////////////////////////////////////////////
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
            else if (arr->isS()) {
                printf("\"%s\", ", arr->e<std::string>(i).c_str());
            }
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
                    printf(", ");
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
        restCount = ShapeUtils::getNumOfSubArrs(arr->shapeInfo(), {0});
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

//////////////////////////////////////////////////////////////////////////
void NDArray::printIndexedBuffer(const char* msg, Nd4jLong limit) const {

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
            // todo do we need this
            // printf("\"%lld\"\n", this->getOffset(e));
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
    fflush(stdout);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void* NDArray::templatedPointerShift(const Nd4jLong offset) const {
    return const_cast<T*>(reinterpret_cast<T const*>(buffer()) + offset);
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void* NDArray::templatedPointerShift, (const Nd4jLong offset) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected
NDArray NDArray::transpose() const &{
    NDArray newArr(getDataBuffer(), ShapeDescriptor(shapeInfo()), getContext(), bufferOffset());
    newArr.transposei();

    return newArr;
}

//////////////////////////////////////////////////////////////////////////
// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected
NDArray NDArray::transpose() && {

    this->transposei();
    return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected
void NDArray::transpose(NDArray& target) const {

    auto correctShape = ShapeUtils::evalTranspShapeInfo(*this, getContext()->getWorkspace());
    if(!shape::equalsStrict(correctShape, target.shapeInfo()))
        throw std::runtime_error("NDArray::transpose method: the shapeInfo of target array is wrong !");

    target._buffer = _buffer;
    target._offset = _offset;
    target._isView = true;
}

////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed
void NDArray::transposei() {
    std::vector<int> perm;
    for (int e = this->rankOf() - 1; e >= 0; e--)
        perm.emplace_back(e);

    this->permutei(perm);
}

////////////////////////////////////////////////////////////////////////
bool NDArray::equalsTo(const NDArray &other, double eps) const {
    return equalsTo(&other, eps);
}

//////////////////////////////////////////////////////////////////////////
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
bool NDArray::reshapei(const char order, const std::initializer_list<Nd4jLong>& shape, const bool copyToNewBuff) {
    std::vector<Nd4jLong> vShape(shape);
    return reshapei(order, vShape, copyToNewBuff);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::reshapei(const std::initializer_list<Nd4jLong>& shape, const bool copyToNewBuff) {
    return reshapei(ordering(), shape, copyToNewBuff);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::reshapei(const std::vector<Nd4jLong>& shape, const bool copyToNewBuff) {
    return reshapei(ordering(), shape, copyToNewBuff);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::enforce(const std::initializer_list<Nd4jLong> &dimensions, char order) {
    std::vector<Nd4jLong> dims(dimensions);
    enforce(dims, order);
}

//////////////////////////////////////////////////////////////////////////
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

    char order = o == 'a' ? this->ordering() : o;
    setShapeInfo(ShapeDescriptor(dataType(), order, dimensions));
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
    }
    else
        throw std::runtime_error("Not implemented yet");
}

//////////////////////////////////////////////////////////////////////////
// create new array with corresponding order and shape, new array will point to the same _buffer as this array
NDArray NDArray::reshape(const char order, const std::vector<Nd4jLong>& shape, const bool copyToNewBuff) const & {

    NDArray newArr(getDataBuffer(), ShapeDescriptor(shapeInfo()), getContext(), bufferOffset());
    newArr.reshapei(order, shape, copyToNewBuff);

    return newArr;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reshape(const char order, const std::vector<Nd4jLong>& shape, const bool copyToNewBuff) && {

    this->reshapei(order, shape, copyToNewBuff);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tilei(const std::vector<Nd4jLong>& reps) {
    *this = this->tile(reps);
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::sizeAt(const int dim) const {

    if (dim >= this->rankOf() || dim < -this->rankOf())
        throw std::runtime_error("NDArray::sizeAt: bad size index requested");

    if (dim >= 0)
        return shape::shapeOf(_shapeInfo)[dim];
    else
        return shape::shapeOf(_shapeInfo)[this->rankOf() + dim];
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::strideAt(const int dim) const {

    if (dim >= this->rankOf() || dim < -this->rankOf())
        throw std::runtime_error("NDArray::strideAt: Bad size index requested");

    if (dim >= 0)
        return shape::stride(_shapeInfo)[dim];
    else
        return shape::stride(_shapeInfo)[this->rankOf() + dim];
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(const std::initializer_list<int>& dimensions) {
    std::vector<int> vec(dimensions);
    return permutei(vec);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(const std::vector<int>& dimensions) {
    return permutei(dimensions.data(), rankOf());
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

    return permutei(ivec.data(), rankOf());
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const int* dimensions, const int rank) const & {

    // evaluate shapeInfo for output (permuted) array ret
    auto shapeInfoPermuted = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, getContext()->getWorkspace());
    NDArray ret(getDataBuffer(), ShapeDescriptor(shapeInfoPermuted), getContext(), bufferOffset());
	ret._isView = true;
    return ret;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const int* dimensions, const int rank) && {

    this->permutei(dimensions, rank);
    return std::move(*this);
}

/////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const Nd4jLong* dimensions, const int rank) const &{
    int tempDims[MAX_RANK];
    shape::convertT<Nd4jLong, int>(const_cast<Nd4jLong *>(dimensions), tempDims, rank);
    return permute(tempDims, rank);
}

/////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const Nd4jLong* dimensions, const int rank) && {

    this->permutei(dimensions, rank);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::vector<int>& dimensions) const &{

    return permute(dimensions.data(), rankOf());
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::vector<int>& dimensions) && {

    this->permutei(dimensions);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::vector<Nd4jLong>& dimensions) const & {

    return permute(dimensions.data(), rankOf());
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::vector<Nd4jLong>& dimensions) && {

    this->permutei(dimensions);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::initializer_list<int>& dimensions) const &{

    std::vector<int> vec(dimensions);
    return permute(vec);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::initializer_list<int>& dimensions) && {

    this->permutei(dimensions);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::initializer_list<Nd4jLong>& dimensions) const & {
    std::vector<Nd4jLong> vec(dimensions);
    return permute(vec);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::permute(const std::initializer_list<Nd4jLong>& dimensions) && {

    this->permutei(dimensions);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::permute(const int* dimensions, const int rank, NDArray& target) const {
    if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
        throw std::runtime_error("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

    auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target.getContext()->getWorkspace());

    target.setShapeInfo(shapeInfoNew);
    target._buffer = _buffer;
    target._offset = _offset;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::permute(const Nd4jLong *dimensions, const int rank, NDArray& target) const {
    if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
        throw std::runtime_error("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

    auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, target.getContext()->getWorkspace());

    target.setShapeInfo(shapeInfoNew);
    target._buffer = _buffer;
    target._offset = _offset;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::permute(const std::vector<int>& dimensions, NDArray& target) const {
    permute(dimensions.data(), rankOf(), target);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::permute(const std::vector<Nd4jLong>& dimensions, NDArray& target) const {
    permute(dimensions.data(), rankOf(), target);
}

//////////////////////////////////////////////////////////////////////////
// check whether array is identity matrix
bool NDArray::isIdentityMatrix() {
    if (isS())
        throw std::runtime_error("NDArray::isIdentityMatrix: you can't use this method on String array!");
	if(rankOf() !=2 || rows() != columns())
	   throw std::runtime_error("isIdentityMatrix method: matrix must be square and have rank = 2 !");

	const double eps = 1e-5f;
	for(Nd4jLong i=0; i<rows(); ++i)
	   if(sd::math::nd4j_abs(e<double>(i,i) - 1.f) > eps)
		  return false;

	for(Nd4jLong i=0; i<rows(); ++i) {
        for(Nd4jLong j=0; j<columns(); ++j) {
            if (i == j)
                continue;
            if(sd::math::nd4j_abs(e<double>(i,j)) > eps)
                return false;
		}
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
    auto trMul = MmulHelper::mmul(this, &tr, nullptr, 1.f, 0.f);

    bool result = trMul->isIdentityMatrix();
    delete trMul;

    return result;
}

//////////////////////////////////////////////////////////////////////////
template <>
const std::string* ND4J_EXPORT NDArray::bufferAsT() const {
    throw std::runtime_error("This method is NOT supposed to be used");
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
const T* NDArray::bufferAsT() const {
    // FIXME: do we REALLY want sync here?
    syncToHost();

    return reinterpret_cast<const T*>(buffer());
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT const, * NDArray::bufferAsT() const, LIBND4J_TYPES);

template <typename T>
T* NDArray::bufferAsT() {
    syncToHost();
    return reinterpret_cast<T*>(buffer());
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT, * NDArray::bufferAsT(), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
NDArray NDArray::subarray(IndicesList& idx) const {

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
    return NDArray((*this)(indexes, true, true));
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::subarray(const std::initializer_list<NDIndex*>& idx) const {

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

    return NDArray((*this)(indexes, true, true));
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::subarray(const Intervals& idx) const {

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

    return NDArray((*this)(indexes, true));
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArray::asT() const{

    auto result = isScalar() ? NDArray('c', {}, std::vector<double>{0.}, DataTypeUtils::fromT<T>(), this->getContext()) : NDArray(ordering(), getShapeAsVector(), DataTypeUtils::fromT<T>(), this->getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformAny(getContext(), transform::AnyOps::Assign, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT NDArray NDArray::asT, () const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArray::asS() const {

    if (!isS())
        throw std::runtime_error("NDArray::asS: you can use this method only for String array!");

    auto dtype = DataTypeUtils::fromT<T>();

    if (!(DataTypeUtils::isS(dtype)))
        throw std::invalid_argument("NDArray::asS: invalid DataType used");

    if (dtype == dataType()) {

        Nd4jLong offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
        const auto nInputoffsets = bufferAsT<Nd4jLong>();
        std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(offsetsLength + nInputoffsets[lengthOf()], dtype, getContext()->getWorkspace(), true);

        NDArray res(pBuffer, ShapeDescriptor(dtype, ordering(), getShapeAsVector()), getContext());
        res.setAttached(getContext()->getWorkspace() != nullptr);

        preparePrimaryUse({ &res }, { this });
        memcpy(res.bufferAsT<int8_t>(), nInputoffsets, offsetsLength);
        auto data = res.bufferAsT<int8_t>() + offsetsLength;
        const auto inData = bufferAsT<int8_t>() + offsetsLength;
        memcpy(data, inData, nInputoffsets[lengthOf()]);

        registerPrimaryUse({ &res }, { this });
        return res;
    }

    Nd4jLong offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());

    std::vector<Nd4jLong> offsets(lengthOf() + 1);

    const auto nInputoffsets = bufferAsT<Nd4jLong>();

    Nd4jLong start = 0, stop = 0;
    Nd4jLong dataLength = 0;

    auto data = bufferAsT<int8_t>() + offsetsLength;
    for (Nd4jLong e = 0; e < lengthOf(); e++) {
        offsets[e] = dataLength;
        start = nInputoffsets[e];
        stop = nInputoffsets[e + 1];
        if (dataType() == DataType::UTF8) {
            dataLength +=  (dtype == DataType::UTF16) ? unicode::offsetUtf8StringInUtf16(data + start, stop)
                        : unicode::offsetUtf8StringInUtf32(data + start, stop);
        }
        else if (dataType() == DataType::UTF16) {
            dataLength += (dtype == DataType::UTF32) ? unicode::offsetUtf16StringInUtf32(data + start, (stop / sizeof(char16_t)) )
                        : unicode::offsetUtf16StringInUtf8(data + start, (stop / sizeof(char16_t)));
        }
        else {
            dataLength += (dtype == DataType::UTF16) ? unicode::offsetUtf32StringInUtf16(data + start, (stop / sizeof(char32_t)))
                        : unicode::offsetUtf32StringInUtf8(data + start, (stop / sizeof(char32_t)));
        }
    }
    offsets[lengthOf()] = dataLength;

    std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(offsetsLength + dataLength, dtype, getContext()->getWorkspace(), true);

    NDArray res(pBuffer, ShapeDescriptor(dtype, ordering(), getShapeAsVector()), getContext());
    res.setAttached(getContext()->getWorkspace() != nullptr);

    preparePrimaryUse({ &res }, { this });

    memcpy(res.bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

    auto outData = res.bufferAsT<int8_t>() + offsetsLength;
    const auto inData = bufferAsT<int8_t>() + offsetsLength;

    auto func = PRAGMA_THREADS_FOR{
        for (int e = start; e < stop; e++) {
           auto cdata = outData + offsets[e];
           auto end = nInputoffsets[e + 1];
           auto idata = inData + nInputoffsets[e];
           if (dtype == DataType::UTF16) {
               if (dataType() == DataType::UTF8) {
                   unicode::utf8to16(idata, outData, end);
               }
               else {
                   unicode::utf32to16(idata, outData, (end / sizeof(char32_t)));
               }
           }
           else if (dtype == DataType::UTF32) {
               if (dataType() == DataType::UTF8) {
                   unicode::utf8to32(idata, cdata, end);
               }
               else {
                   unicode::utf16to32(idata, outData, (end / sizeof(char16_t)));
               }
           }
           else {
               if (dataType() == DataType::UTF16) {
                   unicode::utf16to8(idata, outData, (end / sizeof(char16_t)));
               }
               else {
                   unicode::utf32to8(idata, outData, (end / sizeof(char32_t)));
               }
           }
        }
    };

    samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

    registerPrimaryUse({ &res }, { this });

    return res;
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT NDArray NDArray::asS, () const, LIBND4J_STRINGTYPES);

////////////////////////////////////////////////////////////////////////
NDArray NDArray::asT(DataType dtype) const {

    if (isS() && !DataTypeUtils::isS(dtype))
        throw std::runtime_error("NDArray::asT: you can't use this method on String array with not string DataType!");

    if (!isS() && DataTypeUtils::isS(dtype))
        throw std::runtime_error("NDArray::asT: you can't use this method on not String array with string DataType!");

    if (isS()){
        BUILD_SINGLE_SELECTOR(dtype, return asS, (), LIBND4J_STRINGTYPES);
    } else {
        BUILD_SINGLE_SELECTOR(dtype, return asT, (), LIBND4J_TYPES);
    }

    return NDArray();
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::cast(DataType dtype) const {

    if (isS() && !DataTypeUtils::isS(dtype))
        throw std::runtime_error("NDArray::cast: you can't use this method on String array with not string DataType!");

    if (!isS() && DataTypeUtils::isS(dtype))
        throw std::runtime_error("NDArray::cast: you can't use this method on not String array with string DataType!");

    return this->asT(dtype);
}

////////////////////////////////////////////////////////////////////////
void NDArray::cast(NDArray& target, DataType dtype) {
    if (isS())
        throw std::runtime_error("NDArray::cast: you can't use this method on String array!");
    // TODO: to be implemented properly
    target.assign(this);
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator+=(const NDArray& other) {

    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");
    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL))
        throw sd::datatype_exception::build("NDArray operator+=: Cannot add different types", this->dataType(), other.dataType());

    if (this->lengthOf() != 1 && other.lengthOf() == 1) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execScalar(getContext(), sd::scalar::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execPairwiseTransform(getContext(), sd::pairwise::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else{
        const Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, getContext()->getWorkspace()))
            throw std::invalid_argument("NDArray::operator+=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(shapeInfo(), bShape)) {
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), other, *this, false);
        }
        else {
            NDArray result(bShape, true, getContext());
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), other, result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator-=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator-=: you can't use this method on String array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL))
        throw sd::datatype_exception::build("NDArray operator-=: Cannot subtract different types", this->dataType(), other.dataType());

    if (lengthOf() != 1 && other.lengthOf() == 1) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execScalar(getContext(), sd::scalar::Subtract, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execPairwiseTransform(getContext(), sd::pairwise::Subtract, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else{
        const Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, getContext()->getWorkspace()))
            throw std::invalid_argument("NDArray::operator-=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(shapeInfo(), bShape)) {
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), other, *this, false);
        }
        else {
            NDArray result(bShape, true, getContext());
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), other, result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator*=(const NDArray& other) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");
    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType() && (this->dataType() != DataType::BOOL || other.dataType() != BOOL))
        throw sd::datatype_exception::build("NDArray operator*=: Cannot multiply different types", this->dataType(), other.dataType());

    if (lengthOf() != 1 && other.lengthOf() == 1) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execScalar(getContext(), sd::scalar::Multiply, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execPairwiseTransform(getContext(), sd::pairwise::Multiply, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else{
        const Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, getContext()->getWorkspace()))
            throw std::invalid_argument("NDArray::operator*=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), other, *this, false);
        }
        else {
            NDArray result(bShape, true, getContext());
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), other, result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator/=(const NDArray& other) {
    if (isS() || other.isS())
        throw std::runtime_error("NDArray::operator/=: you can't use this method on String array!");
    if (other.isB())
        throw std::runtime_error("NDArray::operator/=: you can't divide by bool array!");

    if (!Environment::getInstance()->isExperimentalBuild() && this->dataType() != other.dataType()) {
        throw sd::datatype_exception::build("NDArray operator/=: Cannot divide different types", this->dataType(), other.dataType());
    }

    if (lengthOf() != 1 && other.lengthOf() == 1) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execScalar(getContext(), sd::scalar::Divide, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
        NDArray::prepareSpecialUse({this}, {this, &other});
        NativeOpExecutioner::execPairwiseTransform(getContext(), sd::pairwise::Divide, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({this}, {this, &other});
    }
    else{
        const Nd4jLong *bShape = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, bShape, getContext()->getWorkspace()))
            throw std::invalid_argument("NDArray::operator/=: the shapes of this and other arrays are not suitable for broadcast operation !");

        if(shape::equalsTypesAndShapesSoft(_shapeInfo, bShape)) {
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), other, *this, false);
        }
        else {
            NDArray result(bShape, true, getContext());
            this->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), other, result, false);
            *this = std::move(result);      // move assignment operator, zero cost copy
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::operator+=(const T value) {
    if (isS())
        throw std::runtime_error("NDArray::operator+=: you can't use this method on String array!");

    auto other = NDArrayFactory::create(this->dataType(), value, getContext());

    NDArray::prepareSpecialUse({this}, {&other});

    NativeOpExecutioner::execScalar(getContext(), sd::scalar::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({this}, {});
}
template ND4J_EXPORT void NDArray::operator+=(const double value);
template ND4J_EXPORT void NDArray::operator+=(const float value);
template ND4J_EXPORT void NDArray::operator+=(const float16 value);
template ND4J_EXPORT void NDArray::operator+=(const bfloat16 value);
template ND4J_EXPORT void NDArray::operator+=(const Nd4jLong value);
template ND4J_EXPORT void NDArray::operator+=(const int value);
template ND4J_EXPORT void NDArray::operator+=(const bool value);

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::operator-=(const T value) {
    if (isS())
        throw std::runtime_error("NDArray::operator-=: you can't use this method on String array!");

    auto other = NDArrayFactory::create(dataType(), value, getContext());

    NDArray::prepareSpecialUse({this}, {&other});

    NativeOpExecutioner::execScalar(getContext(), sd::scalar::Subtract, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({this}, {});
}
template ND4J_EXPORT void NDArray::operator-=(const double value);
template ND4J_EXPORT void NDArray::operator-=(const float value);
template ND4J_EXPORT void NDArray::operator-=(const float16 value);
template ND4J_EXPORT void NDArray::operator-=(const bfloat16 value);
template ND4J_EXPORT void NDArray::operator-=(const Nd4jLong value);
template ND4J_EXPORT void NDArray::operator-=(const int value);
template ND4J_EXPORT void NDArray::operator-=(const bool value);

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::operator*=(const T scalar) {
    if (isS())
        throw std::runtime_error("NDArray::operator*=: you can't use this method on String array!");

    auto other = NDArrayFactory::create(this->dataType(), scalar, getContext());
    NDArray::prepareSpecialUse({this}, {&other});
    NativeOpExecutioner::execScalar(getContext(), sd::scalar::Multiply, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);

    NDArray::registerSpecialUse({this}, {});
}
template ND4J_EXPORT void NDArray::operator*=(const double scalar);
template ND4J_EXPORT void NDArray::operator*=(const float scalar);
template ND4J_EXPORT void NDArray::operator*=(const float16 scalar);
template ND4J_EXPORT void NDArray::operator*=(const bfloat16 scalar);
template ND4J_EXPORT void NDArray::operator*=(const Nd4jLong scalar);
template ND4J_EXPORT void NDArray::operator*=(const int scalar);
template ND4J_EXPORT void NDArray::operator*=(const int16_t scalar);
template ND4J_EXPORT void NDArray::operator*=(const int8_t scalar);
template ND4J_EXPORT void NDArray::operator*=(const uint8_t scalar);
template ND4J_EXPORT void NDArray::operator*=(const bool scalar);

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::operator/=(const T scalar) {
    if (isS())
        throw std::runtime_error("NDArray::operator/=: you can't use this method on String array!");

    auto other = NDArrayFactory::create(this->dataType(), scalar, getContext());
    NDArray::prepareSpecialUse({this}, {&other});
    NativeOpExecutioner::execScalar(getContext(), sd::scalar::Divide, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({this}, {});
}
template ND4J_EXPORT void NDArray::operator/=(const double scalar);
template ND4J_EXPORT void NDArray::operator/=(const float scalar);
template ND4J_EXPORT void NDArray::operator/=(const float16 scalar);
template ND4J_EXPORT void NDArray::operator/=(const bfloat16 scalar);
template ND4J_EXPORT void NDArray::operator/=(const Nd4jLong scalar);
template ND4J_EXPORT void NDArray::operator/=(const int scalar);
template ND4J_EXPORT void NDArray::operator/=(const int16_t scalar);
template ND4J_EXPORT void NDArray::operator/=(const int8_t scalar);
template ND4J_EXPORT void NDArray::operator/=(const uint8_t scalar);
template ND4J_EXPORT void NDArray::operator/=(const bool scalar);

////////////////////////////////////////////////////////////////////////
// negative operator, it makes all array elements = -elements
NDArray NDArray::operator-() const & {
    if (isS())
        throw std::runtime_error("NDArray::negative-: you can't use this method on String array!");

    NDArray result(shapeInfo(), false, getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformSame(getContext(), sd::transform::Neg, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::operator-() && {
    if (isS())
        throw std::runtime_error("NDArray::negative-: you can't use this method on String array!");

    NDArray::prepareSpecialUse({this}, {this});
    NativeOpExecutioner::execTransformSame(getContext(), sd::transform::Neg, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {this});

    return std::move(*this);
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
void NDArray::tileToShape(const std::vector<Nd4jLong>& shape, NDArray& target) {
    if(&target != this) {
        this->tile(target);
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
void NDArray::tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray& target) {
    tileToShape(std::vector<Nd4jLong>(shape), target);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::tileToShape(const Nd4jLong* shapeInfo) {

    NDArray result(const_cast<Nd4jLong*>(shapeInfo), false, getContext());
    tile(result);
    return result;
}

////////////////////////////////////////////////////////////////////////
double NDArray::getTrace() const {
    if (isS())
        throw std::runtime_error("NDArray::getTrace: you can't use this method on String array!");

    int  rank    = rankOf();
    auto shape   = shapeOf();
    int  minDim  = 100000000;

    Nd4jLong indices[MAX_RANK];
    for(int j = 0; j < rank; ++j)
        indices[j] = 1;

    auto offset = shape::getOffset(shapeInfo(), indices);

    for(int i = 0; i < rank; ++i)
        if(minDim > shape[i])
            minDim = shape[i];

    double sum = 0.;

    for(int i = 0; i < minDim; ++i)
        sum += e<double>(i * offset);

    return sum;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::quantize(const NDArray& array) {

    if(!array.isR())
        throw std::invalid_argument("NDArray::quantize: type of array should be from real space!");

    auto ws = array.getContext()->getWorkspace();

    Nd4jLong* shapeInfo = ShapeBuilders::copyShapeInfo(array.shapeInfo(), true, ws);
    ArrayOptions::setPropertyBit(shapeInfo, ARRAY_QUANTIZED);

    std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(TypeCast::estimateQuantizedSize(array.lengthOf()), ArrayOptions::dataType(shapeInfo), ws);

    NDArray result(buffer, ShapeDescriptor(shapeInfo), array.getContext());

    return result;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, const NDArray& other, NDArray& target, const bool checkTargetShape, ExtraArguments *extraArgs) const {

    if (isS())
        throw std::runtime_error("NDArray::applyTrueBroadcast: you can't use this method on String array!");

    if(((op.s == scalar::Divide || op.s == scalar::FloorDiv || op.s == scalar::FloorMod) && other.isB()) || (op.s == scalar::ReverseDivide && this->isB()))
        throw std::runtime_error("NDArray::applyTrueBroadcast method: you can't divide by bool array !");

    if (isEmpty() || other.isEmpty())
        return;

    // if (lengthOf() == 1) {
    //     target.assign(this);
    //     target.applyPairwiseTransform(op.p, other, extraArgs);
    //     return;
    // }
    // if (other.lengthOf() == 1) {
    //     const_cast<NDArray*>(this)->applyScalarArr(op.s, other, target, extraArgs);
    //     return;
    // }

    if(checkTargetShape) {
        const Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        if(!shape::equalsTypesAndShapesSoft(target.shapeInfo(), newShapeInfo))
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape or type of target array is wrong !");
    }

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace());
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace());
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcast(getContext(), op.b, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastBoolOpsTuple op, const NDArray& other, NDArray& target, const bool checkTargetShape, ExtraArguments *extraArgs) const {

    if (isS())
        throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");

    if (isEmpty() || other.isEmpty())
        return;

    // if (lengthOf() == 1) {
    //     NDArray temp(target._shapeInfo, dataType(), false, getContext());
    //     temp.assign(this);
    //     temp.applyPairwiseTransform(op.p, other, target,  extraArgs);
    //     return;
    // }
    // if (other.lengthOf() == 1) {
    //     this->applyScalarArr(op.s, other, target, extraArgs);
    //     return;
    // }

    if(checkTargetShape) {
        const Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        if(!shape::equalsSoft(target._shapeInfo, newShapeInfo) || target.dataType() != DataType::BOOL)
            throw std::runtime_error("NDArray::applyTrueBroadcast bool method: the shape or type of target array is wrong !");
        if(dataType() != other.dataType())
            throw std::invalid_argument("NDArray::applyTrueBroadcast bool method: this and other arrays must have the same type !");
    }

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace());
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace());
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcastBool(getContext(), op.b, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr);
    registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastIntOpsTuple op, const NDArray& other, NDArray& target, const bool checkTargetShape, ExtraArguments *extraArgs) const {

    if (isS())
        throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");

    if (isEmpty() || other.isEmpty())
        return;

    // if (lengthOf() == 1) {
    //     NDArray temp(target._shapeInfo, dataType(), false, getContext());
    //     temp.assign(this);
    //     temp.applyPairwiseTransform(op.p, other, target,  extraArgs);
    //     return;
    // }
    // if (other.lengthOf() == 1) {
    //     this->applyScalarArr(op.s, other, target, extraArgs);
    //     return;
    // }

    if(checkTargetShape) {
        const Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, false, newShapeInfo, getContext()->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        if(!shape::equalsSoft(target._shapeInfo, newShapeInfo) || target.dataType() != this->dataType())
            throw std::runtime_error("NDArray::applyTrueBroadcast int method: the shape or type of target array is wrong !");
        if(dataType() != other.dataType())
            throw std::invalid_argument("NDArray::applyTrueBroadcast int method: this and other arrays must have the same type !");
    }

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace());
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace());
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcastInt(getContext(), op.b, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs) const & {
    if (isEmpty() || other.isEmpty()) {
        if (isEmpty())
            return NDArray(*this);
        else
            return NDArray(other);
    }

    const Nd4jLong* newShapeInfo = nullptr;
    if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of new array = max->rankOf)()
        throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
    NDArray result(newShapeInfo, true, getContext());

    this->applyTrueBroadcast(op, other, result, false, extraArgs);

    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, NDArray&& other, ExtraArguments *extraArgs) const & {
    if (isEmpty() || other.isEmpty()) {
        if (isEmpty())
            return NDArray(*this);
        else
            return NDArray(other);
    }

    const Nd4jLong* newShapeInfo = nullptr;
    if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of new array = max->rankOf)()
        throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");

    if(!shape::shapeEquals(newShapeInfo, other.shapeInfo())) {

        NDArray result(newShapeInfo, true, getContext());
        this->applyTrueBroadcast(op, other, result, false, extraArgs);
        return std::move(result);
    }

    this->applyTrueBroadcast(op, other, other, false, extraArgs);
    return std::move(other);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs) && {
    if (isEmpty() || other.isEmpty()) {
        if (isEmpty())
            return NDArray(*this);
        else
            return NDArray(other);
    }

    const Nd4jLong* newShapeInfo = nullptr;
    if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of new array = max->rankOf)()
        throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");

    if(!shape::shapeEquals(newShapeInfo, shapeInfo())) {

        NDArray result(newShapeInfo, true, getContext());
        this->applyTrueBroadcast(op, other, result, false, extraArgs);
        return std::move(result);
    }

    this->applyTrueBroadcast(op, other, *this, false, extraArgs);
    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, NDArray&& other, ExtraArguments *extraArgs) && {
    if (isEmpty() || other.isEmpty()) {
        if (isEmpty())
            return NDArray(*this);
        else
            return NDArray(other);
    }

    const Nd4jLong* newShapeInfo = nullptr;
    if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, getContext()->getWorkspace()))          // the rank of new array = max->rankOf)()
        throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");

    const bool thisMove  = shape::shapeEquals(newShapeInfo, shapeInfo());
    const bool otherMove = shape::shapeEquals(newShapeInfo, other.shapeInfo());

    if(!thisMove && !otherMove) {

        NDArray result(newShapeInfo, true, getContext());
        this->applyTrueBroadcast(op, other, result, false, extraArgs);
        return std::move(result);
    }

    if(thisMove) {
        this->applyTrueBroadcast(op, other, *this, false, extraArgs);
        return std::move(*this);
    }

    // otherMove
    this->applyTrueBroadcast(op, other, other, false, extraArgs);
    return std::move(other);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::Ops op, const std::vector<int>& dimensions, const NDArray& other, NDArray& target, ExtraArguments* extraArgs) {

    if (dimensions.size() == 0)
        return;

    if (isS())
        throw std::runtime_error("NDArray::applyBroadcast: you can't use this method on String array!");
    if(((op == broadcast::Divide || op == broadcast::FloorDiv || op == broadcast::FloorMod) && other.isB()) || (op == broadcast::ReverseDivide && this->isB()))
        throw std::runtime_error("NDArray::applyBroadcast: you can't divide by array!");
    if(isEmpty() || other.isEmpty()) {
        if(!target.isEmpty())
            throw std::runtime_error("NDArray::applyBroadcast method: when some of input arrays (or both) is empty, target array must be empty as well !");
        return;
    }

    // if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
    //     NDArray::prepareSpecialUse({&target}, {this, &other});
    //     NativeOpExecutioner::execPairwiseTransform(getContext(), fromBroadcastToPairwise(op), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr);
    //     NDArray::registerSpecialUse({&target}, {this, &other});
    //     return;
    // }

    if(target.dataType() != DataTypeUtils::pickPairwiseResultType(shapeInfo(), other.shapeInfo()))
        throw std::invalid_argument("NDArray::applyBroadcast method: wrong type of target array !");
    if(!target.isSameShape(this) && !target.isSameShape(other))
        throw std::invalid_argument("NDArray::applyBroadcast method: one of of two input arrays (this or other) should has the same shape as target array!");

    std::vector<int> copy(dimensions);

    if (dimensions.size() > 1)
        std::sort(copy.begin(), copy.end());

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace(), copy);
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace(), copy);
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcast(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::BoolOps op, const std::vector<int>& dimensions, const NDArray& other, NDArray& target, ExtraArguments* extraArgs) {

    if (dimensions.size() == 0)
        return;

    if (isS())
        throw std::runtime_error("NDArray::applyBroadcast BoolOps: you can't use this method on String array!");
    if(isEmpty() || other.isEmpty()) {
        if(!target.isEmpty())
            throw std::runtime_error("NDArray::applyBroadcast BoolOps: when some of input arrays (or both) is empty, target array must be empty as well !");
        return;
    }

    // if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
    //     NDArray::prepareSpecialUse({&target}, {this, &other});
    //     NativeOpExecutioner::execPairwiseBoolTransform(getContext(), fromBroadcastToPairwiseBool(op), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr);
    //     NDArray::registerSpecialUse({&target}, {this, &other});
    //     return;
    // }

    if(target.dataType() != DataType::BOOL)
        throw std::invalid_argument("NDArray::applyBroadcast bool method: type of target array must be BOOL!");
    if(!target.isSameShape(this) && !target.isSameShape(other))
        throw std::invalid_argument("NDArray::applyBroadcast bool method: one of of two input arrays (this or other) should has the same shape as target array!");
    if(_dataType != other._dataType)
        throw std::invalid_argument("NDArray::applyBroadcast bool method: this and other arrays must have the same type !");

    std::vector<int> copy(dimensions);

    if (dimensions.size() > 1)
        std::sort(copy.begin(), copy.end());

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace(), copy);
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace(), copy);
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcastBool(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr);
    registerSpecialUse({&target}, {this, &other});
}


//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::IntOps op, const std::vector<int>& dimensions, const NDArray& other, NDArray& target, ExtraArguments* extraArgs) {

    if (dimensions.empty())
        return;

    if (!isZ())
        throw std::runtime_error("NDArray::applyBroadcast IntOps: you can't use this method on non-Integer array!");
    if(isEmpty() || other.isEmpty()) {
        if(!target.isEmpty())
            throw std::runtime_error("NDArray::applyBroadcast IntOps: when some of input arrays (or both) is empty, target array must be empty as well !");
        return;
    }

    // if (other.lengthOf() == lengthOf() && this->rankOf() == other.rankOf()) {
    //     NDArray::prepareSpecialUse({&target}, {this, &other});
    //     NativeOpExecutioner::execPairwiseIntTransform(getContext(), fromBroadcastToPairwiseInt(op), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr);
    //     NDArray::registerSpecialUse({&target}, {this, &other});
    //     return;
    // }

    if(target.dataType() != dataType())
        throw std::invalid_argument("NDArray::applyBroadcast int method: type of target array must be the same as input!");
    if(!target.isSameShape(this) && !target.isSameShape(other))
        throw std::invalid_argument("NDArray::applyBroadcast int method: one of of two input arrays (this or other) should has the same shape as target array!");
    if(_dataType != other._dataType)
        throw std::invalid_argument("NDArray::applyBroadcast int method: this and other arrays must have the same type !");

    std::vector<int> copy(dimensions);

    if (dimensions.size() > 1)
        std::sort(copy.begin(), copy.end());

    Nd4jLong const* xShapeInfoH = shapeInfo();
    Nd4jLong const* yShapeInfoH = other.shapeInfo();
    Nd4jLong const* xShapeInfoD = specialShapeInfo();
    Nd4jLong const* yShapeInfoD = other.specialShapeInfo();

    if(!isSameShape(target)) {
        auto xPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), shapeInfo(), getContext()->getWorkspace(), copy);
        xShapeInfoH = reinterpret_cast<Nd4jLong const*>(xPack.primary());
        xShapeInfoD = reinterpret_cast<Nd4jLong const*>(xPack.special());
    }
    if(!other.isSameShape(target)) {
        auto yPack = ConstantShapeHelper::getInstance()->createShapeInfoWithUnitiesForBroadcast(target.shapeInfo(), other.shapeInfo(), other.getContext()->getWorkspace(), copy);
        yShapeInfoH = reinterpret_cast<Nd4jLong const*>(yPack.primary());
        yShapeInfoD = reinterpret_cast<Nd4jLong const*>(yPack.special());
    }

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execBroadcastInt(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD, other.buffer(), yShapeInfoH, other.specialBuffer(), yShapeInfoD, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::Ops op, const std::initializer_list<int> dimensions, const NDArray& tadArray, NDArray& target, ExtraArguments* extraArgs) {
    std::vector<int> vec(dimensions);
    applyBroadcast(op, vec, tadArray, target, extraArgs);
}

////////////////////////////////////////////////////////////////////////
void* NDArray::operator new(size_t i) {
    if (sd::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
        sd::memory::Workspace* ws = sd::memory::MemoryRegistrator::getInstance()->getWorkspace();
        return ws->allocateBytes((Nd4jLong) i);
    }
    else {
        auto p = malloc(i);
        CHECK_ALLOC(p, "Failed to allocate new NDArray", i);
        return p;
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator delete(void* p) {
    if (!sd::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached())
        free(p);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> NDArray::asVectorT() {

    std::vector<T> result(this->lengthOf());

    PRAGMA_OMP_SIMD
    for (int e = 0; e < this->lengthOf(); e++)
        result[e] = this->e<T>(e);

    return result;
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT std::vector, NDArray::asVectorT(), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length
bool NDArray::reshapei(const char order, const std::vector<Nd4jLong>& cshape, const bool copyToNewBuff) {

    // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary
    if(order == ordering() && shape::shapeEquals(rankOf(), shapeOf(), cshape.size(), cshape.data()))
        return true;

    const bool isOutShapeEmpty = std::find(cshape.begin(), cshape.end(), 0) != cshape.end();

    if(isEmpty() && !isOutShapeEmpty)
        throw std::invalid_argument("NDArray::reshapei: can't reshape empty array to non-empty !");
    if(!isEmpty() && isOutShapeEmpty)
        throw std::invalid_argument("NDArray::reshapei: can't reshape non-empty array to empty !");
    if(isEmpty() && isOutShapeEmpty) {
        Nd4jLong* shapeInfoNew = ShapeBuilders::emptyShapeInfo(dataType(), order, cshape, getContext()->getWorkspace());
        setShapeInfo(shapeInfoNew);
        RELEASE(shapeInfoNew, getContext()->getWorkspace());
        return true;
    }

    std::vector<Nd4jLong> shape(cshape);
    int rank = shape.size();

    // looking for negative in shape

    int numberNegativesOnes = 0;

    Nd4jLong* shape_ = shape.data();
    for (int i = 0; i < (int) shape.size(); i++) {
        if (shape[i] < 0) {
            if (numberNegativesOnes >= 1)
                throw std::runtime_error("NDArray::reshapei: only one dimension can be negative at once");

            numberNegativesOnes++;

            int shapeLength = 1;
            for (int j = 0; j < (int) shape.size(); j++)
                if (i != j)
                    shapeLength *= shape_[j];

            Nd4jLong realShape = sd::math::nd4j_abs<int>(lengthOf() / shapeLength);
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

    Nd4jLong arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(platformBuffer() == nullptr || arrLength != this->lengthOf()) {
        this->printShapeInfo("Mismatched shape");
        sd::Logger::printv("Shape requested: ", shape);
        nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
        throw std::runtime_error("NDArray::reshapei: bad input shape!");
    }

    Nd4jLong *shapeInfoNew;
    ALLOCATE(shapeInfoNew, getContext()->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

    bool canReshape = shape::reshapeC(shapeInfo(), order, shape.size(), shape.data(), shapeInfoNew);

    if (canReshape) {
        setShapeInfo(shapeInfoNew);
    }
    else {
        NDArray temp(order, shape, dataType(), getContext());
        if(copyToNewBuff)
            this->applyTransform(transform::Assign, temp, nullptr);
        *this = std::move(temp);
    }

    RELEASE(shapeInfoNew, getContext()->getWorkspace());

    return canReshape;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::nullify() {
    if (isEmpty())
        return;

    if (isView() || ews() != 1)
        assign(0);
    else
        _buffer->setToZeroBuffers();

}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, sd::DataType dtype, const void *value) {
    BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, sd::DataType dtype, const void *value), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(sd::pairwise::Ops op, const NDArray& other, NDArray& target, ExtraArguments *extraParams) const{
    if (isS())
        throw std::runtime_error("NDArray::applyPairwiseTransform: you can't use this method on String array!");
    if (other.lengthOf() != target.lengthOf())
        throw std::invalid_argument("NDArray::applyPairwiseTransform method - lengths of arrays are mismatched");
    if (target.dataType() != this->dataType() && target.dataType() != other.dataType())
        throw std::invalid_argument("NDArray::applyPairwiseTransform method - type of target array must be the same as type of this or other array !");

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execPairwiseTransform(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr);
    NDArray::registerSpecialUse({&target}, {this, &other});

    if (extraParams != nullptr)
        synchronize("NDArray::applyPairwiseTransform");
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(sd::pairwise::BoolOps op, const NDArray& other, NDArray& target, ExtraArguments *extraParams) const{
    if (isS())
        throw std::runtime_error("NDArray::applyPairwiseTransform BoolOps: you can't use this method on String array!");
    if (other.lengthOf() != target.lengthOf())
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - lengths of arrays are mismatched");
    if (!target.isB())
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - result must have bool type");
    if (dataType() != other.dataType())
        throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - this and other arrays must have the same type !");

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execPairwiseBoolTransform(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr);
    NDArray::registerSpecialUse({&target}, {this, &other});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(sd::pairwise::IntOps op, const NDArray& other, NDArray& target, ExtraArguments *extraParams) const{
    if (isS())
        throw std::runtime_error("NDArray::applyPairwiseTransform IntOps: you can't use this method on String array!");
    if (other.lengthOf() != target.lengthOf())
        throw std::invalid_argument("NDArray::applyPairwiseTransform IntOps method - lengths of arrays are mismatched");
    if (!target.isZ())
        throw std::invalid_argument("NDArray::applyPairwiseTransform IntOps method - result must have bool type");
    if (dataType() != other.dataType())
        throw std::invalid_argument("NDArray::applyPairwiseTransform IntOps method - this and other arrays must have the same type !");

    NDArray::prepareSpecialUse({&target}, {this, &other});
    NativeOpExecutioner::execPairwiseIntTransform(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr);
    NDArray::registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(sd::pairwise::Ops op, const NDArray& other, ExtraArguments *extraParams) {
    applyPairwiseTransform(op, other, *this, extraParams);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void NDArray::templatedDoubleAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
    auto x = reinterpret_cast<X *>(xBuffer);
    const auto y = reinterpret_cast<const Y *>(yBuffer);
    x[xOffset] = static_cast<X>(y[yOffset]);
}
BUILD_DOUBLE_TEMPLATE(template ND4J_EXPORT void NDArray::templatedDoubleAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES, LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::varianceAlongDimension(sd::variance::Ops op, NDArray& target, const bool biasCorrected, const std::vector<int>& dimensions) const {

    if (isS())
        throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

    if (!target.isR())
        throw std::runtime_error("NDArray::varianceAlongDimension: target array must have FLOAT type");

    NDArray::prepareSpecialUse({&target}, {this});

    if(rankOf() == dimensions.size() || dimensions.empty())
        NativeOpExecutioner::execSummaryStatsScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), biasCorrected);
    else {
        std::vector<int> copy(dimensions);
        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimensions);
        NativeOpExecutioner::execSummaryStats(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), pDims, dimensions.size(), packX.platformShapeInfo(), packX.platformOffsets(), biasCorrected);
        synchronize("NDArray::varianceAlongDimension");
    }

    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceAlongDimension(sd::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
    if (isS())
        throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

    std::vector<int> copy(dimensions);
    if (copy.size() > 1)
        std::sort(copy.begin(), copy.end());

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(dataType()), false, false, getContext()->getWorkspace());
    NDArray result(newShape, true, getContext());

    this->varianceAlongDimension(op, result, biasCorrected, dimensions);

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceAlongDimension(sd::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
    return varianceAlongDimension(op, biasCorrected, std::vector<int>(dimensions));
}

////////////////////////////////////////////////////////////////////////
void NDArray::varianceAlongDimension(sd::variance::Ops op, NDArray &target, const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
    varianceAlongDimension(op, target, biasCorrected, std::vector<int>(dimensions));
}

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
NDArray NDArray::dup(const char newOrder) const {

    if (isEmpty())
        return NDArrayFactory::empty(dataType(), getContext());

    char order = newOrder == 'a' ? ordering() : newOrder;

    // for now string arrays require special treatment
    if (isS()) {
        if (dataType() == DataType::UTF8) {
            std::vector<std::string> strings(lengthOf());

            auto func = PRAGMA_THREADS_FOR{
                    for (auto i = start; i < stop; i++) {
                           strings[i] = std::move(this->e<std::string>(i));
                    }
            };

            samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

            return NDArray(getShapeAsVector(), strings, dataType(), getContext());
        }
        if (dataType() == DataType::UTF16) {
            std::vector<std::u16string> strings(lengthOf());

            auto func = PRAGMA_THREADS_FOR{
                    for (auto i = start; i < stop; i++) {
                           strings[i] = std::move(this->e<std::u16string>(i));
                    }
            };

            samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

            return NDArray(getShapeAsVector(), strings, dataType(), getContext());
        }

        std::vector<std::u32string> strings(lengthOf());
        auto func = PRAGMA_THREADS_FOR{
               for (auto i = start; i < stop; i++) {
                      strings[i] = std::move(this->e<std::u32string>(i));
               }
        };

        samediff::Threads::parallel_for(func, 0, lengthOf(), 1);

        return NDArray(getShapeAsVector(), strings, dataType(), getContext());
    }

    NDArray result(order, isScalar() ? std::vector<Nd4jLong>({0}) : getShapeAsVector(), dataType(), getContext());
    result.assign(*this);

    return result;
}

////////////////////////////////////////////////////////////////////////
// This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
bool NDArray::equalsTo(const NDArray *other, double eps) const {

    if (dataType() != other->dataType() || lengthOf() != other->lengthOf())
        return false;

    // we need to be able to compare [1, len] to [len]
    if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
        // FIXME: do something here?
    } else if (!shape::equalsSoft(shapeInfo(), other->shapeInfo()))
        return false;

    if (isS()) {
        // string is special case, we'll compare them one by one, considering both arrays are guaranteed to have the same length

        if (dataType() == DataType::UTF8) {
            for (Nd4jLong e = 0; e < this->lengthOf(); e++) {
                auto s1 = this->e<std::string>(e);
                auto s2 = other->e<std::string>(e);

                if (s1 != s2)
                    return false;
            }
        }
        else if (dataType() == DataType::UTF16) {
            for (Nd4jLong e = 0; e < this->lengthOf(); e++) {
                auto s1 = this->e<std::u16string>(e);
                auto s2 = other->e<std::u16string>(e);

                if (s1 != s2)
                    return false;
            }
        }
        else {
            for (Nd4jLong e = 0; e < this->lengthOf(); e++) {
                auto s1 = this->e<std::u32string>(e);
                auto s2 = other->e<std::u32string>(e);

                if (s1 != s2)
                    return false;
            }
        }

        return true;
    } else {
        // regular numeric types
        NDArray tmp(sd::DataType::FLOAT32, getContext()); // scalar = 0

        ExtraArguments extras({0.0, 0.0, eps});

        NDArray::prepareSpecialUse({&tmp}, {this, other});
        NativeOpExecutioner::execReduce3Scalar(getContext(), reduce3::EqualsWithEps, buffer(), shapeInfo(),
                                               specialBuffer(), specialShapeInfo(),
                                               extras.argumentsAsT(DataType::FLOAT32), other->buffer(),
                                               other->shapeInfo(), other->specialBuffer(),
                                               other->specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(),
                                               tmp.specialBuffer(), tmp.specialShapeInfo());
        NDArray::registerSpecialUse({&tmp}, {this, other});

        synchronize("NDArray::equalsTo");

        if (tmp.e<Nd4jLong>(0) != 0)
            return false;

        return true;
    }
}

//////////////////////////////////////////////////////////////////////////
template <>
std::string NDArray::e(const Nd4jLong i) const {

    if (!isS())
        throw std::runtime_error("Can't get std::string out of non-string array");

    if (i == lengthOf())
        throw std::runtime_error("Can't get std::string for index out of range");


    if (this->dataType() == DataType::UTF16) {
        auto u16 = this->e<std::u16string>(i);
        std::string s;
        StringUtils::u16StringToU8String(u16, s);
        return s;
    }

    if (this->dataType() == DataType::UTF32) {
        auto u32 = this->e<std::u32string>(i);
        std::string s;
        StringUtils::u32StringToU8String(u32, s);
        return s;
    }

    NDArray::preparePrimaryUse({}, {this});

    auto offsets = bufferAsT<Nd4jLong>();
    auto offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
    auto start = offsets[i];
    auto end = offsets[i + 1];
    auto data = bufferAsT<int8_t>() + offsetsLength + start;

    std::string r(reinterpret_cast<const char*>(data), (end - start));

    registerPrimaryUse({}, {this});

    return r;
}

template <>
std::u16string NDArray::e(const Nd4jLong i) const {

    if (!isS())
        throw std::runtime_error("Can't get std::u16string out of non-string array");

    if(i == lengthOf())
        throw std::runtime_error("Can't get std::u16string for index out of range");

    if (this->dataType() == DataType::UTF8) {
        auto u = this->e<std::string>(i);
        std::u16string s;
        StringUtils::u8StringToU16String(u, s);
        return s;
    }

    if (this->dataType() == DataType::UTF32) {
        auto u32 = this->e<std::u32string>(i);
        std::u16string s;
        StringUtils::u32StringToU16String(u32, s);
        return s;
    }

    NDArray::preparePrimaryUse({}, { this });

    auto offsets = bufferAsT<Nd4jLong>();
    Nd4jLong offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
    Nd4jLong start = offsets[i];
    Nd4jLong end = offsets[i + 1];
    auto data = bufferAsT<int8_t>() + offsetsLength + start;

    std::u16string r(reinterpret_cast<const char16_t*>(data), (end - start) / sizeof(char16_t));

    registerPrimaryUse({}, { this });

    return r;
}

template <>
std::u32string NDArray::e(const Nd4jLong i) const {

    if (!isS())
        throw std::runtime_error("Can't get std::u32string out of non-string array");

    if (i == lengthOf())
        throw std::runtime_error("Can't get std::u32string for index out of range");

    if (this->dataType() == DataType::UTF8) {
        auto u = this->e<std::string>(i);
        std::u32string s;
        StringUtils::u8StringToU32String(u, s);
        return s;
    }

    if (this->dataType() == DataType::UTF16) {
        auto u16 = this->e<std::u16string>(i);
        std::u32string s;
        StringUtils::u16StringToU32String(u16, s);
        return s;
    }

    NDArray::preparePrimaryUse({}, { this });

    auto offsets = bufferAsT<Nd4jLong>();
    Nd4jLong offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
    Nd4jLong start = offsets[i];
    Nd4jLong end = offsets[i + 1];

    auto data = bufferAsT<int8_t>() + offsetsLength + start;

    std::u32string r(reinterpret_cast<const char32_t*>(data), (end - start) / sizeof(char32_t));

    registerPrimaryUse({}, { this });

    return r;
}

//////////////////////////////////////////////////////////////////////////
template <>
utf8string NDArray::e(const Nd4jLong i) const {

    if (!isS())
        throw std::runtime_error("This method is available for String arrays only");

    auto rp = getOffset(i);

    syncToHost();
    tickReadHost();

    return *(reinterpret_cast<utf8string* const*>(buffer())[rp]);
}

/////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::e(const Nd4jLong i) const {

    const auto rp = getOffset(i);

    NDArray::preparePrimaryUse({}, {this});
    NDArray::registerPrimaryUse({}, {this});
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), rp), LIBND4J_TYPES);

}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT , NDArray::e(const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
template <typename T>
T NDArray::e(const Nd4jLong i, const Nd4jLong j) const {

    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
        throw std::invalid_argument("NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !");

    const Nd4jLong coords[2] = {i, j};
    const auto xOffset = shape::getOffset(shapeInfo(), coords);

    NDArray::preparePrimaryUse({}, {this});
    NDArray::registerPrimaryUse({}, {this});

    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), LIBND4J_TYPES);

    return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT , NDArray::e(const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
template <typename T>
T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {

    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
        throw std::invalid_argument("NDArray::e(i,j,k): one of input indexes is out of array length or rank!=3 !");

    const Nd4jLong coords[3] = {i, j, k};
    const auto xOffset = shape::getOffset(shapeInfo(), coords);

    NDArray::preparePrimaryUse({}, {this});
    NDArray::registerPrimaryUse({}, {this});

    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), LIBND4J_TYPES);

    return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
template <typename T>
T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l) const {

    if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
        throw std::invalid_argument("NDArray::e(i,j,k,l): one of input indexes is out of array length or rank!=4 !");

    const Nd4jLong coords[4] = {i, j, k, l};
    const auto xOffset = shape::getOffset(shapeInfo(), coords);

    NDArray::preparePrimaryUse({}, {this});
    NDArray::registerPrimaryUse({}, {this});

    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), LIBND4J_TYPES);

    return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template ND4J_EXPORT , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::e(const Nd4jLong i) const {

    const auto offset = getOffset(i);

    NDArray scalar(dataType(), getContext());

    scalar.copyBuffersContinuouslyFrom(*this, sizeOfT(), 0, bufferOffset() + offset);

    return scalar;
}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
void NDArray::applyTransform(sd::transform::FloatOps op, NDArray& target, ExtraArguments *extraParams) {

    if (isS())
        throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

    if (!target.isR())
        throw std::runtime_error("NDArray::applyTransform FloatOps: target array must have one of FLOAT types");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execTransformFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::AnyOps op, NDArray& target, ExtraArguments *extraParams) {

    if (isS())
        throw std::runtime_error("NDArray::applyTransform AnyOps: you can't use this method on String array!");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execTransformAny(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::SameOps op, NDArray& target, ExtraArguments *extraParams) {

    if (isS())
        throw std::runtime_error("NDArray::applyTransform SameOps: you can't use this method on String array!");

    if (target.dataType() != dataType())
        throw std::runtime_error("NDArray::applyTransform SameOps: target array must have the same data type as original array");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execTransformSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::StrictOps op, NDArray& target, ExtraArguments *extraParams) {
    if (isS())
        throw std::runtime_error("NDArray::applyTransform StrictOps: you can't use this method on String array!");

    if (!this->isR() || !target.isR() || (this->dataType() != target.dataType()))
        throw std::runtime_error("NDArray::applyTransform StrictOps: both Source and Target array must have same FLOAT type !");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execTransformStrict(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::BoolOps op, NDArray& target, ExtraArguments *extraParams) {
    if (isS())
        throw std::runtime_error("NDArray::applyTransform BoolOps: you can't use this method on String array!");

    if (!target.isB())
        throw std::runtime_error("NDArray::applyTransform BoolOps: target array must have one of BOOL types");

    NDArray::prepareSpecialUse({&target}, {this});
    NativeOpExecutioner::execTransformBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()) : nullptr, nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::FloatOps op, void *extraParams) const & {
    if (isS())
        throw std::runtime_error("NDArray::transform FloatOps: you can't use this method on String array!");

    NDArray result(ordering(), getShapeAsVector(), DataTypeUtils::pickFloatingType(dataType()), getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::FloatOps op, void *extraParams) && {
    if (isS())
        throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

    NDArray::prepareSpecialUse({this}, {this});
    NativeOpExecutioner::execTransformFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {this});

    return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::SameOps op, void *extraParams) const & {
    if (isS())
        throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

    NDArray result(shapeInfo(), false, getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::SameOps op, void *extraParams) && {
    if (isS())
        throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

    NDArray::prepareSpecialUse({this}, {this});
    NativeOpExecutioner::execTransformSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {this});

    return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::StrictOps op, void *extraParams) const & {
    if (!this->isR())
        throw std::runtime_error("Source array must have one of FLOAT types");

    NDArray result(shapeInfo(), false, getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformStrict(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::StrictOps op, void *extraParams) && {
    if (!this->isR())
        throw std::runtime_error("Source array must have one of FLOAT types");

    NDArray::prepareSpecialUse({this}, {this});
    NativeOpExecutioner::execTransformStrict(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {this});

    return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::BoolOps op, void *extraParams) const & {
    if (isS())
        throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

    NDArray result(ordering(), getShapeAsVector(), sd::DataType::BOOL, getContext());

    NDArray::prepareSpecialUse({&result}, {this});
    NativeOpExecutioner::execTransformBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({&result}, {this});

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::BoolOps op, void *extraParams) && {
    if (isS())
        throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

    NDArray::prepareSpecialUse({this}, {this});
    NativeOpExecutioner::execTransformBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams, nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {this});

    return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::Ops op, const NDArray& scalar, NDArray& target, ExtraArguments *extraParams) {
    if (isS())
        throw std::runtime_error("NDArray::applyScalarArr: you can't use this method on String array!");
    if (scalar.lengthOf() != 1)
        throw std::invalid_argument("NDArray::applyScalarArr method: operand is not a scalar!");

    if(target.dataType() != DataTypeUtils::pickPairwiseResultType(shapeInfo(), scalar.shapeInfo()) && !(target.dataType() == dataType() || target.dataType() == scalar.dataType()))
        throw std::invalid_argument("NDArray::applyScalarArr method: wrong type of target array!");

    NDArray::prepareSpecialUse({&target}, {this, &scalar});
    NativeOpExecutioner::execScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), scalar.buffer(), scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()): nullptr);
    NDArray::registerSpecialUse({&target}, {this, &scalar});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::BoolOps op, const NDArray& scalar, NDArray &target, ExtraArguments *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::applyScalarArr BoolOps: you can't use this method on String array!");
    if (!target.isB())
        throw std::invalid_argument("NDArray::applyScalarArr bool method: target has not bool type!");
    if (dataType() != scalar.dataType()) {
        nd4j_printf("NDArray::applyScalarArr BoolOps: this dtype: [%i]; scalar dtype: [%i]\n", this->dataType(), scalar.dataType());
        throw std::invalid_argument("NDArray::applyScalarArr bool method: this and scalar arrays must have the same type!");
    }

    NDArray::prepareSpecialUse({&target}, {this, &scalar});
    NativeOpExecutioner::execScalarBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), scalar.buffer(), scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()): nullptr);
    NDArray::registerSpecialUse({&target}, {this, &scalar});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::IntOps op, const NDArray& scalar, NDArray &target, ExtraArguments *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::applyScalarArr IntOps: you can't use this method on String array!");

    if (target.dataType() != this->dataType())
        throw std::invalid_argument("NDArray::applyScalarArr int method: target has not bool type!");
    if (dataType() != scalar.dataType()) {
        nd4j_printf("NDArray::applyScalarArr IntOps: this dtype: [%i]; scalar dtype: [%i]\n", this->dataType(), scalar.dataType());
        throw std::invalid_argument("NDArray::applyScalarArr int method: this and scalar arrays must have the same type!");
    }

    NDArray::prepareSpecialUse({&target}, {this, &scalar});
    NativeOpExecutioner::execScalarInt(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), scalar.buffer(), scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target.dataType()): nullptr);
    NDArray::registerSpecialUse({&target}, {this, &scalar});
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::IntOps op, const T scalar, NDArray& target, ExtraArguments *extraParams) const {

    NDArray scalarArr = NDArrayFactory::create(this->dataType(), scalar, getContext());
    applyScalarArr(op, scalarArr, target, extraParams);
}

template <> ND4J_EXPORT void NDArray::applyScalar(sd::scalar::IntOps op, const NDArray& scalar, NDArray &target, ExtraArguments *extraParams) const { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
template ND4J_EXPORT void NDArray::applyScalar<double>(sd::scalar::IntOps op, const double scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<float>(sd::scalar::IntOps op, const float scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<float16>(sd::scalar::IntOps op, const float16 scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<bfloat16>(sd::scalar::IntOps op, const bfloat16 scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<Nd4jLong>(sd::scalar::IntOps op, const Nd4jLong scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int>(sd::scalar::IntOps op, const int scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int16_t>(sd::scalar::IntOps op, const int16_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int8_t>(sd::scalar::IntOps op, const int8_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<uint8_t>(sd::scalar::IntOps op, const uint8_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<bool>(sd::scalar::IntOps op, const bool scalar, NDArray &target, ExtraArguments *extraParams) const;

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::Ops op, const T scalar, NDArray& target, ExtraArguments *extraParams) {

    auto scalarArr = NDArrayFactory::create<T>(dataType(), scalar, this->getContext());
    applyScalarArr(op, scalarArr, target, extraParams);
}
template <> ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const NDArray& scalar, NDArray &target, ExtraArguments *extraParams) { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const double scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const float scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const float16 scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const bfloat16 scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const Nd4jLong scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int16_t scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int8_t scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const uint8_t scalar, NDArray &target, ExtraArguments *extraParams);
template ND4J_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const bool scalar, NDArray &target, ExtraArguments *extraParams);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::BoolOps op, const T scalar, NDArray &target, ExtraArguments *extraParams) const {

    NDArray scalarArr = NDArrayFactory::create<T>(scalar, getContext());
    applyScalarArr(op, scalarArr, target, extraParams);
}

template <> ND4J_EXPORT void NDArray::applyScalar(sd::scalar::BoolOps op, const NDArray& scalar, NDArray &target, ExtraArguments *extraParams) const { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
template ND4J_EXPORT void NDArray::applyScalar<double>(sd::scalar::BoolOps op, const double scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<float>(sd::scalar::BoolOps op, const float scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<float16>(sd::scalar::BoolOps op, const float16 scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<bfloat16>(sd::scalar::BoolOps op, const bfloat16 scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<Nd4jLong>(sd::scalar::BoolOps op, const Nd4jLong scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int>(sd::scalar::BoolOps op, const int scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int16_t>(sd::scalar::BoolOps op, const int16_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<int8_t>(sd::scalar::BoolOps op, const int8_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<uint8_t>(sd::scalar::BoolOps op, const uint8_t scalar, NDArray &target, ExtraArguments *extraParams) const;
template ND4J_EXPORT void NDArray::applyScalar<bool>(sd::scalar::BoolOps op, const bool scalar, NDArray &target, ExtraArguments *extraParams) const;

////////////////////////////////////////////////////////////////////////
void NDArray::applyIndexReduce(sd::indexreduce::Ops op, NDArray& target, const std::vector<int>& dimensions, const ExtraArguments *extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

    if (target.dataType() != sd::DataType::INT64 && target.dataType() != sd::DataType::INT32)
        throw std::runtime_error("NDArray::applyIndexReduce operations return INT32/INT64");

    void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

    NDArray::prepareSpecialUse({&target}, {this});

    if (target.lengthOf() == 1) {
        NativeOpExecutioner::execIndexReduceScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }
    else {
        std::vector<int> copy = dimensions;
        shape::checkDimensions(rankOf(), copy);
        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(shapeInfo(), copy);
        NativeOpExecutioner::execIndexReduce(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets());
        synchronize("NDArray::applyIndexReduce");
    }

    registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
// reduce dimensions in this array relying on index operations
NDArray NDArray::applyIndexReduce(sd::indexreduce::Ops op, const std::vector<int>& dimensions, const ExtraArguments* extraParams ) const {

    std::vector<int> copy = dimensions;
    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::INT64, false, false, getContext()->getWorkspace());
    NDArray result(newShape, true, getContext());

    applyIndexReduce(op, result, copy, extraParams);

    return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 operations to this and other array, return result in new output array
NDArray NDArray::applyReduce3(sd::reduce3::Ops op, const NDArray& other, const ExtraArguments* extraParams) const {

    if (isS())
        throw std::runtime_error("NDArray::applyReduce3 method: you can't use this method on String array!");
    if(dataType() != other.dataType())
        throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");
    // check shapes consistency
    if(!isSameShape(other))
        throw std::runtime_error("NDArray::applyReduce3 method: the shapes of this and other arrays must be the same !");
    // create shapeInfo for scalar
    auto newShape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(dataType()), getContext()->getWorkspace());
    // create output array (scalar)
    NDArray result(newShape, true, getContext());
    RELEASE(newShape, getContext()->getWorkspace());
    // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
    void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(dataType()) : nullptr;

    NDArray::prepareSpecialUse({&result}, {this, &other});
    NativeOpExecutioner::execReduce3Scalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    NDArray::registerSpecialUse({&result}, {this, &other});

    return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 (exec) operations to this and other array, return result in new output array
NDArray NDArray::applyReduce3(sd::reduce3::Ops op, const NDArray& other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {

    if (isS())
        throw std::runtime_error("NDArray::applyReduce3: you can't use this method on String array!");
    if(dataType() != other.dataType())
        throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);
    shape::checkDimensions(other.rankOf(), copy);

    auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(dataType()), false, false, getContext()->getWorkspace());
    NDArray result(newShape, true, getContext());
    // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
    void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(dataType()) : nullptr;

    NDArray::prepareSpecialUse({&result}, {this, &other});

    // perform calculations
    if(rankOf() == copy.size() && other.rankOf() == copy.size()) {
        NativeOpExecutioner::execReduce3Scalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo());
    }
    else {

        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;

        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(shapeInfo(), copy);
        auto packY = sd::ConstantTadHelper::getInstance()->tadForDimensions(other.shapeInfo(), copy);

        if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()) || (packX.numberOfTads() != packY.numberOfTads() && packX.numberOfTads() != 1 && packY.numberOfTads() != 1))
            throw std::runtime_error("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

        NativeOpExecutioner::execReduce3(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets(), packY.platformShapeInfo(), packY.platformOffsets());
    }

    registerSpecialUse({&result}, {this, &other});

    return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 (execAll) operations to this and other array, return result in new output array
NDArray NDArray::applyAllReduce3(sd::reduce3::Ops op, const NDArray& other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {
    if (isS())
        throw std::runtime_error("NDArray::applyAllReduce3: you can't use this method on String array!");
    if(dataType() != other.dataType())
        throw std::runtime_error("NDArray::applyAllReduce3 method: the types of this and other arrays must be the same !");

    // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);
    shape::checkDimensions(other.rankOf(), copy);

    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(shapeInfo(), copy);
    auto packY = ConstantTadHelper::getInstance()->tadForDimensions(other.shapeInfo(), copy);

    // check tads shapes
    if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()))
        throw std::runtime_error("NDArray::applyAllReduce3 method: the shapes of array tads are different !");

    // set newShape for output array
    auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(DataTypeUtils::pickFloatingType(dataType()), 'c', {packX.numberOfTads(), packY.numberOfTads()});

    // create output array
    NDArray result(newShape, true, getContext());

    // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
    void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(dataType()) : nullptr;

    auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;

    NDArray::prepareSpecialUse({&result}, {this, &other});
    NativeOpExecutioner::execReduce3All(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other.buffer(), other.shapeInfo(), other.specialBuffer(), other.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets(), packY.platformShapeInfo(), packY.platformOffsets());
    NDArray::registerSpecialUse({&result}, {this, &other});

    return result;
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::FloatOps op, NDArray& target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: you can't use this method on String array!");
    if (!target.isR())
        throw std::invalid_argument("NDArray::reduceAlongDimension FloatOps: requires target array to be present and have type form real space!");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target.ordering(), copy, *this, keepDims, supportOldShapes, getContext()->getWorkspace());
        if(!shape::shapeEquals(newShape, target.shapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: wrong target shape!");
    }

    NDArray::prepareSpecialUse({&target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }
    else {
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(shapeInfo(), copy);
        NativeOpExecutioner::execReduceFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), copy.data(), copy.size(), packX.platformShapeInfo(), packX.platformOffsets());
    }
    synchronize("NDArray::reduceAlongDimension FloatOps");

    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::SameOps op, NDArray& target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: you can't use this method on String array!");
    if (target.dataType() != dataType())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps: requires target array to be present and have same dtype as input");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target.ordering(), copy, *this, keepDims, supportOldShapes, getContext()->getWorkspace());
        if(!shape::shapeEquals(newShape, target.shapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension SameOps: wrong target shape!");
    }

    NDArray::prepareSpecialUse({&target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }
    else { //if (!isEmpty()) {
        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), copy);
        NativeOpExecutioner::execReduceSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets());
    }
    synchronize("NDArray::reduceAlongDimension SameOps");

    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::LongOps op, NDArray& target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps: you can't use this method on String array!");
    if (target.dataType() != DataType::INT64)
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps: requires target array to be present and have type of INT64");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target.ordering(), copy, *this, keepDims, supportOldShapes, getContext()->getWorkspace());
        if(!shape::shapeEquals(newShape, target.shapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension LongOps: wrong target shape!");
    }

    NDArray::prepareSpecialUse({&target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }
    else {
        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), copy);
        NativeOpExecutioner::execReduceLong(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets());
    }
    synchronize("NDArray::reduceAlongDimension LongOps");

    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::BoolOps op, NDArray& target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: you can't use this method on String array!");
    if (!target.isB())
        throw std::invalid_argument("NDArray::reduceAlongDimension BoolOps cuda: requires target array to be present and have BOOL type!");

    std::vector<int> copy(dimensions);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target.ordering(), copy, *this, keepDims, supportOldShapes, getContext()->getWorkspace());
        if(!shape::shapeEquals(newShape, target.shapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({&target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {
        NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo());
    }
    else {
        auto pDims = sd::Environment::getInstance()->isCPU() ? copy.data() : nullptr;
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), copy);
        NativeOpExecutioner::execReduceBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), pDims, copy.size(), packX.platformShapeInfo(), packX.platformOffsets());
    }
    synchronize("NDArray::reduceAlongDimension LongOps");

    NDArray::registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
template <typename T>
void NDArray::p(const Nd4jLong i, const T value) {

    if (i >= lengthOf())
        throw std::invalid_argument("NDArray::p(i, value): input index is out of array length !");

    auto rp = getOffset(i);
    const void *pV = reinterpret_cast<const void*>(const_cast<T *>(&value));

    NDArray::preparePrimaryUse({this}, {}, true);
    BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), templatedSet<, T>(this->buffer(), rp, pV), LIBND4J_TYPES);
    NDArray::registerPrimaryUse({this}, {});
}

template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const double value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const float value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const float16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const bfloat16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const int value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const int8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const uint8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const uint16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const uint32_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const uint64_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const int16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const bool value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j
template <typename T>
void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {

    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
        throw std::invalid_argument("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

    void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
    Nd4jLong coords[2] = {i, j};
    auto xOffset = shape::getOffset(shapeInfo(), coords);

    NDArray::preparePrimaryUse({this}, {}, true);
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), LIBND4J_TYPES);
    NDArray::registerPrimaryUse({this}, {});
}
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const double value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bfloat16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint32_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint64_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bool value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
template <typename T>
void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value) {
    //(*this)(i,j,k) = value;
    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
        throw std::invalid_argument("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");

    NDArray::preparePrimaryUse({this}, {}, true);

    void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
    Nd4jLong coords[3] = {i, j, k};
    auto xOffset = shape::getOffset(shapeInfo(), coords);
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), LIBND4J_TYPES);
    NDArray::registerPrimaryUse({this}, {});
}
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const double value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bfloat16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint32_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint64_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bool value);

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value) {
    //(*this)(i,j,k) = value;
    if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
        throw std::invalid_argument("NDArray::p(i,j,k,l, value): one of input indexes is out of array length or rank!=4 !");

    void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
    Nd4jLong coords[4] = {i, j, k, l};
    auto xOffset = shape::getOffset(shapeInfo(), coords);

    NDArray::preparePrimaryUse({this}, {}, true);
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), LIBND4J_TYPES);
    NDArray::registerPrimaryUse({this}, {});
}
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const double value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bfloat16 value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const Nd4jLong value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint8_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint32_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint64_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int16_t value);
template ND4J_EXPORT void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bool value);

////////////////////////////////////////////////////////////////////////
void NDArray::p(const Nd4jLong i, const NDArray& scalar) {

    if(scalar.lengthOf() != 1)
        throw std::invalid_argument("NDArray::p method: input array must be scalar!");
    if (i >= _length)
        throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");

    NDArray::preparePrimaryUse({this}, {&scalar}, true);
    auto rp = getOffset(i);
    BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (buffer(), rp, scalar.dataType(), scalar.buffer()), LIBND4J_TYPES);
    NDArray::registerPrimaryUse({this}, {&scalar});
}

////////////////////////////////////////////////////////////////////////
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const NDArray& scalar) {

        if(scalar.lengthOf() != 1)
            throw std::invalid_argument("NDArray::p method: input array must be scalar!");
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");

//        void *p = reinterpret_cast<void *>(scalar.buffer());
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(shapeInfo(), coords);

        NDArray::preparePrimaryUse({this}, {&scalar}, true);
//        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), LIBND4J_TYPES);
        BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (this->buffer(), xOffset, scalar.dataType(), scalar.buffer()), LIBND4J_TYPES);
        NDArray::registerPrimaryUse({this}, {&scalar});
    }

//////////////////////////////////////////////////////////////////////////
void NDArray::addRowVector(const NDArray& row, NDArray& target) const {

    if (isS())
        throw std::runtime_error("NDArray::addRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || target.rankOf() != 2 || rows() != target.rows() || columns() != target.columns() || !row.isRowVector() || columns() != row.lengthOf())
        throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");
    if(target.dataType() !=  DataTypeUtils::pickPairwiseResultType(dataType(), row.dataType()) && !(isR() && row.isR() && target.isR()))
        throw std::invalid_argument("NDArray::addRowVector: wrong type of target array !");

    int dimension = 1;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({&target}, {this, &row});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), row.buffer(), row.shapeInfo(), row.specialBuffer(), row.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this, &row});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::subRowVector(const NDArray& row, NDArray& target) const {

    if (isS())
        throw std::runtime_error("NDArray::addRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || target.rankOf() != 2 || rows() != target.rows() || columns() != target.columns() || !row.isRowVector() || columns() != row.lengthOf())
        throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");
    if(target.dataType() !=  DataTypeUtils::pickPairwiseResultType(dataType(), row.dataType()) && !(isR() && row.isR() && target.isR()))
        throw std::invalid_argument("NDArray::addRowVector: wrong type of target array !");

    int dimension = 1;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({&target}, {this, &row});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Subtract, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), row.buffer(), row.shapeInfo(), row.specialBuffer(), row.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), &dimension, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this, &row});
}


//////////////////////////////////////////////////////////////////////////
void NDArray::mulRowVector(const NDArray &row, NDArray &target) const {

    if (isS())
        throw std::runtime_error("NDArray::mulRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || target.rankOf() != 2 || rows() != target.rows() || columns() != target.columns() || !row.isRowVector() || columns() != row.columns())
        throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
    if(target.dataType() !=  DataTypeUtils::pickPairwiseResultType(dataType(), row.dataType()))
        throw std::invalid_argument("NDArray::mulRowVector: wrong type of target array !");

    int dimension = 1;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({&target}, {this, &row});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Multiply, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), row.buffer(), row.shapeInfo(), row.specialBuffer(), row.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this, &row});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::divRowVector(const NDArray &row, NDArray &target) const {

    if (isS())
        throw std::runtime_error("NDArray::divRowVector: you can't use this method on String array!");
    if (row.isB())
        throw std::runtime_error("NDArray::divRowVector: you can't divide by bool row!");
    if (rankOf() != 2 || target.rankOf() != 2 || rows() != target.rows() || columns() != target.columns() || !row.isRowVector() || columns() != row.columns())
        throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
    if(target.dataType() !=  DataTypeUtils::pickPairwiseResultType(dataType(), row.dataType()))
        throw std::invalid_argument("NDArray::divRowVector: wrong type of target array !");

    int dimension = 1;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({&target}, {this, &row});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Divide, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), row.buffer(), row.shapeInfo(), row.specialBuffer(), row.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this, &row});
}

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
void NDArray::addiRowVector(const NDArray& row) {

    if (isS())
        throw std::runtime_error("NDArray::addiRowVector: you can't use this method on String array!");
    if (rankOf() != 2 || !row.isRowVector() || columns() != row.lengthOf())
        throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

    int dimension = 1;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({this}, {&row});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), row.buffer(), row.shapeInfo(), row.specialBuffer(), row.specialShapeInfo(), this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {&row});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::addColumnVector(const NDArray &column, NDArray &target) const {
    if (isS())
        throw std::runtime_error("NDArray::addColumnVector: you can't use this method on String array!");
    if (rankOf() != 2 || target.rankOf() != 2 || rows() != target.rows() || columns() != target.columns() || !column.isColumnVector() || rows() != column.lengthOf())
        throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");
    if(target.dataType() !=  DataTypeUtils::pickPairwiseResultType(dataType(), column.dataType()))
        throw std::invalid_argument("NDArray::addColumnVector: wrong type of target array !");

    int dimension = 0;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({&target}, {this, &column});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), column.buffer(), column.shapeInfo(), column.specialBuffer(), column.specialShapeInfo(), target.buffer(), target.shapeInfo(), target.specialBuffer(), target.specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({&target}, {this, &column});
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
void NDArray::addiColumnVector(const NDArray &column) {
    if (isS())
        throw std::runtime_error("NDArray::addiColumnVector: you can't use this method on String array!");
    if (rankOf() != 2 || !column.isColumnVector() || rows() != column.lengthOf())
        throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

    int dimension = 0;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({this}, {&column});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), column.buffer(), column.shapeInfo(), column.specialBuffer(), column.specialShapeInfo(), this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {&column});
}

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
void NDArray::muliColumnVector(const NDArray& column) {
    if (isS())
        throw std::runtime_error("NDArray::muliColumnVector: you can't use this method on String array!");
    if (rankOf() != 2 || !column.isColumnVector() || rows() != column.lengthOf())
        throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

    int dimension = 0;

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(this->shapeInfo(), dimension);

    NDArray::prepareSpecialUse({this}, {&column});
    NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Multiply, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), column.buffer(), column.shapeInfo(), column.specialBuffer(), column.specialShapeInfo(), this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.platformShapeInfo(), packX.platformOffsets(), nullptr, nullptr);
    NDArray::registerSpecialUse({this}, {&column});
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::templatedAssign(void *xBuffer, Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
    if (xBuffer != nullptr && yBuffer != nullptr)
        *(reinterpret_cast<T*>(xBuffer) + xOffset) = *(reinterpret_cast<const T*>(yBuffer) + yOffset);
}
BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void NDArray::templatedAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES);


//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(const int* dimensions, const int rank) {

    auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, getContext()->getWorkspace());
    setShapeInfo(shapeInfo);

    return true;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

    auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, getContext()->getWorkspace());
    setShapeInfo(shapeInfo);

    return true;
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::multipleTensorsAlongDimension(const std::vector<int> &indices, const std::vector<int> &dimensions) const {
    ResultSet result;

    if (indices.size() == 0)
        return result;

    auto pack = ConstantTadHelper::getInstance()->tadForDimensions(shapeInfo(), const_cast<int*>(dimensions.data()), dimensions.size());

    auto tadLength = shape::length(pack.primaryShapeInfo());
    auto numTads = lengthOf() / tadLength;

    for (auto idx: indices) {
        if (idx >= numTads) {
            nd4j_printf("NDArray::multipleTensorsAlongDimension: index %i is higher then number of TADs: %i\n", idx, numTads);
            throw std::runtime_error("Bad index");
        }

        auto array = new NDArray(getDataBuffer(), ShapeDescriptor(pack.primaryShapeInfo()), getContext(), pack.primaryOffsets()[idx] + bufferOffset());
        result.push_back(array);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allTensorsAlongDimension(const std::initializer_list<int>& dimensions) const {
    return allTensorsAlongDimension(std::vector<int>(dimensions));
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allExamples() const {
    std::vector<int> dimensions(rankOf() - 1);
    for (int e = 1; e < rankOf(); e++)
        dimensions[e-1] = e;

    return allTensorsAlongDimension(dimensions);
}

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::getOffset(const Nd4jLong i) const {

    if (i >= lengthOf())
        throw std::invalid_argument("NDArray::getOffset: input index is out of array length !");

    return shape::getIndexOffset(i, _shapeInfo);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::like() {

    return NDArray(shapeInfo(), this->dataType(), false, getContext());
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::ulike() const{

    return NDArray(this, false, getContext());
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::diagonal(const char type) const {

    if (isS())
        throw std::runtime_error("NDArray::diagonal: you can't use this method on String array!");

    const char order = ordering();
    const int  rank  = rankOf();
    Nd4jLong *outShapeInfo;
    ALLOCATE(outShapeInfo, getContext()->getWorkspace(), 8, Nd4jLong);
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

        auto step = shape::getOffset(shapeInfo(), indices);

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

    NDArray result(_buffer, ShapeDescriptor(outShapeInfo), getContext(), bufferOffset());

    RELEASE(outShapeInfo, getContext()->getWorkspace());

    return result;
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allTensorsAlongDimension(const std::vector<int> &dimensions) const {

    ResultSet result;

    if(dimensions.size() == 0)
        return result;

    if(dimensions.back() >= rankOf())
        throw std::runtime_error("NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input array !");


    auto pack = ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, const_cast<int*>(dimensions.data()), dimensions.size());
    auto numTads = pack.numberOfTads();

    for (Nd4jLong idx = 0; idx < numTads; idx++ ) {
        auto array = new NDArray(_buffer, ShapeDescriptor(pack.primaryShapeInfo()), getContext(), pack.primaryOffsets()[idx] + bufferOffset());
        array->_isView = true;
        result.push_back(array);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////
// operator returns sub-array with buffer pointing at this->_buffer + certain offset
NDArray NDArray::operator()(const std::vector<Nd4jLong>& idx, const bool keepUnitiesInShape, const bool isStrided)  const {

    if(isEmpty())
        throw std::invalid_argument("NDArray::operator(sub-arrays): array is empty !");

    // Nd4jLong *outShapeInfo = nullptr;
    //     ALLOCATE(outShapeInfo, workspace, shape::shapeInfoLength(inShapeInfo), Nd4jLong);

    int numOfUntiesInSubArrShape = 0;

    Nd4jLong* subArrShapeInfo = nullptr;

    if(!keepUnitiesInShape) {

        int n(isStrided ? 3 : 2), first, last;

        // calculate the number of unities in shape
        for (uint d = 0; d < rankOf(); ++d) {

            if (idx[n * d] != idx[n * d + 1]) {

                first  = idx[n * d]     >= 0 ? idx[n * d]     : idx[n * d]     + sizeAt(d) + 1;
                last   = idx[n * d + 1] >= 0 ? idx[n * d + 1] : idx[n * d + 1] + sizeAt(d) + 1;
                if(last - first == 1)
                    ++numOfUntiesInSubArrShape;
            }
        }
    }

    ALLOCATE(subArrShapeInfo, getContext()->getWorkspace(), shape::shapeInfoLength(rankOf() - numOfUntiesInSubArrShape), Nd4jLong);

    Nd4jLong offset;

    shape::calcSubArrShapeInfoAndOffset(idx.data(), shapeInfo(), subArrShapeInfo, offset, keepUnitiesInShape, isStrided, numOfUntiesInSubArrShape);

    NDArray result(_buffer, ShapeDescriptor(subArrShapeInfo), getContext(), offset + bufferOffset());
    result._isView = true;

    RELEASE(subArrShapeInfo, getContext()->getWorkspace());

    return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape)  const {

    std::vector<Nd4jLong> idxRanges(2 * rankOf());

    const auto rank = rankOf();
    const auto subArrRank = static_cast<int>(dimsToExclude.size());

    if(subArrRank > rank)
        throw std::invalid_argument("NDArray::operator(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape): static method: dimsToExclude is empty or has size > rank of array !");

    memset(idxRanges.data(), 0, 2 * rank * sizeof(Nd4jLong));

    // subArrRank == 0 means whole array, idxRanges should contain zeros only

    if(subArrRank != 0) {

        std::vector<Nd4jLong> shapeOfSubArr(subArrRank), indexes(subArrRank);
        for(int i = 0; i < subArrRank; ++i)
            shapeOfSubArr[i] = sizeAt(dimsToExclude[i]);

        shape::index2coords(subArrIdx, subArrRank, shapeOfSubArr.data(), indexes.data());

        for(int i = 0; i < subArrRank; ++i) {
            int currIdx = 2 * dimsToExclude[i];
            idxRanges[currIdx]     = indexes[i];
            idxRanges[currIdx + 1] = indexes[i] + 1;
        }
    }

    return (*this)(idxRanges, keepUnitiesInShape);
}

////////////////////////////////////////////////////////////////////////
void NDArray::getSubArrShapeAndOffsets(const std::vector<int>& dimsToExclude, Nd4jLong* &subArrShapeInfo, Nd4jLong* &subArrOffsets, bool keepUnitiesInShape) const {

    if(isEmpty())
        throw std::invalid_argument("NDArray::getSubArrShapeAndOffsets: array is empty !");

    const int rank = rankOf();
    const int subArrRank = (rank == dimsToExclude.size() || keepUnitiesInShape) ? rank : rank - dimsToExclude.size();
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

    // allocate memory
    ALLOCATE(subArrShapeInfo, getContext()->getWorkspace(), shape::shapeInfoLength(subArrRank), Nd4jLong);
    ALLOCATE(subArrOffsets,   getContext()->getWorkspace(), numOfSubArrs, Nd4jLong);

    shape::calcSubArrsShapeInfoAndOffsets(_shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(), subArrShapeInfo, subArrOffsets, keepUnitiesInShape);
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
        _dataType = sd::DataType::INHERIT;
        _shapeInfoD = _shapeInfo = nullptr;
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const Nd4jLong *shapeInfo, const sd::DataType dtype) {

    if (shapeInfo != nullptr) {

        Nd4jLong* shapeInfoTemp = ShapeBuilders::copyShapeInfoAndType(shapeInfo, dtype, true, getContext()->getWorkspace());
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
        _dataType = sd::DataType::INHERIT;
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

//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const ConstantDataBuffer& shapeBuffer) {

    _shapeInfo  = reinterpret_cast<Nd4jLong *>(const_cast<ConstantDataBuffer&>(shapeBuffer).primary());
    #ifdef __CUDABLAS__
    _shapeInfoD = reinterpret_cast<Nd4jLong *>(const_cast<ConstantDataBuffer&>(shapeBuffer).special());
    #endif

    if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
        _length = 0;
    else
        _length = shape::length(_shapeInfo);

    _dataType = ArrayOptions::dataType(_shapeInfo);
}

///////////////////////////////////////////////////////////////////////
// addition operator array + scalar
template <typename T, typename>
NDArray operator+(NDArray&& arr, const T& scalar) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(arr + scalar);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator+(NDArray&& arr, const T& scalar): you can't use this method on String array!");
    if (arr.dataType() != DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()))
       throw std::runtime_error("operator+(NDArray&& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Add, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);
}
template ND4J_EXPORT NDArray operator+(NDArray&& arr, const double& scalar);
template ND4J_EXPORT NDArray operator+(NDArray&& arr, const float& scalar);
template ND4J_EXPORT NDArray operator+(NDArray&& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator+(NDArray&& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator+(NDArray&& arr, const int& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator+(const NDArray& arr, const T& scalar) {

    if (arr.isS())
        throw std::runtime_error("operator+(const NDArray& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Add, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}
template ND4J_EXPORT NDArray operator+(const NDArray& arr, const double& scalar);
template ND4J_EXPORT NDArray operator+(const NDArray& arr, const float& scalar);
template ND4J_EXPORT NDArray operator+(const NDArray& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator+(const NDArray& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator+(const NDArray& arr, const int& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator+(const T& scalar, NDArray&& arr) {
    return  std::move(arr) + scalar;
}
template ND4J_EXPORT NDArray operator+(const double& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator+(const float& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator+(const float16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator+(const bfloat16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator+(const int& scalar, NDArray&& arr);


////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator+(const T& scalar, const NDArray& arr) {
    return arr + scalar;
}
template ND4J_EXPORT NDArray operator+(const double& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator+(const float& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator+(const int& scalar, const NDArray& arr);

///////////////////////////////////////////////////////////////////////
// addition operator array - scalar
template <typename T, typename>
NDArray operator-(NDArray&& arr, const T& scalar) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(arr - scalar);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator-(NDArray&& arr, const T& scalar): you can't use this method on String array!");
    if (arr.dataType() != DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()))
       throw std::runtime_error("operator-(NDArray&& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Subtract, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);
}
template ND4J_EXPORT NDArray operator-(NDArray&& arr, const double& scalar);
template ND4J_EXPORT NDArray operator-(NDArray&& arr, const float& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator-(const NDArray& arr, const T& scalar) {

    if (arr.isS())
        throw std::runtime_error("operator-(const NDArray& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Subtract, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}
template ND4J_EXPORT NDArray operator-(const NDArray& arr, const double& scalar);
template ND4J_EXPORT NDArray operator-(const NDArray& arr, const float& scalar);
template ND4J_EXPORT NDArray operator-(const NDArray& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator-(const NDArray& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator-(const NDArray& arr, const int& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator-(const T& scalar, NDArray&& arr) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(scalar - arr);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator-(const T& scalar, NDArray&& arr): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::ReverseSubtract, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);

}
template ND4J_EXPORT NDArray operator-(const double& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator-(const float& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator-(const float16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator-(const bfloat16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator-(const int& scalar, NDArray&& arr);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator-(const T& scalar, const NDArray& arr) {

    if (arr.isS())
        throw std::runtime_error("operator-(const T& scalar, const NDArray& arr): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::ReverseSubtract, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}
template ND4J_EXPORT NDArray operator-(const double& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator-(const float& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator-(const int& scalar, const NDArray& arr);

///////////////////////////////////////////////////////////////////////
// addition operator array + scalar
template <typename T, typename>
NDArray operator*(NDArray&& arr, const T& scalar) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(arr * scalar);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator*(NDArray&& arr, const T& scalar): you can't use this method on String array!");
    if (arr.dataType() != DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()))
       throw std::runtime_error("operator*(NDArray&& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Multiply, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);
}
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const double& scalar);
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const float& scalar);
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const int& scalar);
template ND4J_EXPORT NDArray operator*(NDArray&& arr, const long long& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator*(const NDArray& arr, const T& scalar) {

    if (arr.isS())
        throw std::runtime_error("operator*(const NDArray& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Multiply, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}

template ND4J_EXPORT NDArray operator*(const NDArray& arr, const double& scalar);
template ND4J_EXPORT NDArray operator*(const NDArray& arr, const float& scalar);
template ND4J_EXPORT NDArray operator*(const NDArray& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator*(const NDArray& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator*(const NDArray& arr, const int& scalar);
template ND4J_EXPORT NDArray operator*(const NDArray& arr, const long long& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator*(const T& scalar, NDArray&& arr) {
    return  std::move(arr) * scalar;
}
template ND4J_EXPORT NDArray operator*(const double& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator*(const float& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator*(const float16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator*(const bfloat16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator*(const int& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator*(const long long& scalar, NDArray&& arr);


////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator*(const T& scalar, const NDArray& arr) {
    return arr * scalar;
}
template ND4J_EXPORT NDArray operator*(const double& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator*(const float& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator*(const float16& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator*(const bfloat16& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator*(const int& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator*(const long long& scalar, const NDArray& arr);

///////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator/(NDArray&& arr, const T& scalar) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(arr / scalar);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator/(NDArray&& arr, const T& scalar): you can't use this method on String array!");
    if (arr.dataType() != DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()))
       throw std::runtime_error("operator/(NDArray&& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Divide, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);
}
template ND4J_EXPORT NDArray operator/(NDArray&& arr, const double& scalar);
template ND4J_EXPORT NDArray operator/(NDArray&& arr, const float& scalar);
template ND4J_EXPORT NDArray operator/(NDArray&& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator/(NDArray&& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator/(NDArray&& arr, const long long& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator/(const NDArray& arr, const T& scalar) {

    if (arr.isS())
        throw std::runtime_error("operator/(const NDArray& arr, const T& scalar): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::Divide, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const double& scalar);
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const float& scalar);
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const float16& scalar);
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const bfloat16& scalar);
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const int& scalar);
template ND4J_EXPORT NDArray operator/(const NDArray& arr, const long long& scalar);

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator/(const T& scalar, NDArray&& arr) {

    if(arr.isView())                        // do not use resources of arrays which use buffers of other original arrays
        return std::move(scalar / arr);     // arr is lvalue inside function body

    if (arr.isS())
        throw std::runtime_error("operator/(const T& scalar, NDArray&& arr): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());

    NDArray::prepareSpecialUse({&arr}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::ReverseDivide, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr}, {&arr, &tmp});

    return std::move(arr);

}
template ND4J_EXPORT NDArray operator/(const double& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator/(const float& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator/(const float16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator/(const bfloat16& scalar, NDArray&& arr);
template ND4J_EXPORT NDArray operator/(const int& scalar, NDArray&& arr);


////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray operator/(const T& scalar, const NDArray& arr) {

    if (arr.isS())
        throw std::runtime_error("operator/(const T& scalar, const NDArray& arr): you can't use this method on String array!");

    auto tmp = NDArrayFactory::create(arr.dataType(), scalar, arr.getContext());
    NDArray result(arr.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr.dataType(), DataTypeUtils::fromT<T>()), false, arr.getContext());

    NDArray::prepareSpecialUse({&result}, {&arr, &tmp});
    NativeOpExecutioner::execScalar(arr.getContext(), sd::scalar::ReverseDivide, arr.buffer(), arr.shapeInfo(), arr.specialBuffer(), arr.specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(), tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {&arr, &tmp});

    return result;
}
template ND4J_EXPORT NDArray operator/(const double& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator/(const float& scalar, const NDArray& arr);
template ND4J_EXPORT NDArray operator/(const int& scalar, const NDArray& arr);

////////////////////////////////////////////////////////////////////////
// addition operator array + array
template <typename T1, typename T2, typename>
NDArray operator+(T1&& arr1, T2&& arr2) {

    if (arr1.isS() || arr2.isS())
        throw std::runtime_error("operator+(T&& arr1, T&& arr2): you can't use this method on String arrays!");
    if (!Environment::getInstance()->isExperimentalBuild() && arr1.dataType() != arr2.dataType() && (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
        throw sd::datatype_exception::build("operator+(T&& arr1, T&& arr2): Cannot multiply different types", arr1.dataType(), arr2.dataType());

    PointersManager pointersManager(arr1.getContext(), "operator+(T&& arr1, T&& arr2)");

    if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {

        const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
        const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

        NDArray* result = nullptr;
        if(isArr1Rvalue)
            result = const_cast<NDArray*>(&arr1);
        else if(isArr2Rvalue)
            result = const_cast<NDArray*>(&arr2);
        else
            result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()), false, arr1.getContext());

        NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
        NativeOpExecutioner::execPairwiseTransform(arr1.getContext(), sd::pairwise::Add, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(), result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({result}, {&arr1, &arr2});

        if(!isArr1Rvalue && !isArr2Rvalue) {
            NDArray res = std::move(*result);
            delete result;
            return std::move(res);
        }

        return std::move(*result);
    }

    return std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), std::forward<T2>(arr2));
}
template ND4J_EXPORT NDArray operator+<NDArray&, NDArray&, void>(NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator+<NDArray&, NDArray, void>(NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator+<NDArray,  NDArray&, void>(NDArray&& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator+<NDArray&, const NDArray&, void>(NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator+<const NDArray&, NDArray&, void>(const NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator+<const NDArray&, NDArray, void>(const NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator+<const NDArray&, const NDArray&, void>(const NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator+<NDArray, const NDArray&, void>(NDArray&& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator+<NDArray, NDArray, void>(NDArray&& arr1, NDArray&& arr2);

////////////////////////////////////////////////////////////////////////
// addition operator array - array
template <typename T1, typename T2, typename>
NDArray operator-(T1&& arr1, T2&& arr2) {

    if (arr1.isS() || arr2.isS())
        throw std::runtime_error("operator-(T&& arr1, T&& arr2): you can't use this method on String arrays!");
    if (!Environment::getInstance()->isExperimentalBuild() && arr1.dataType() != arr2.dataType() && (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
        throw sd::datatype_exception::build("operator-(T&& arr1, T&& arr2): Cannot multiply different types", arr1.dataType(), arr2.dataType());

    PointersManager pointersManager(arr1.getContext(), "operator-(T&& arr1, T&& arr2)");

    if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {

        const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
        const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

        NDArray* result = nullptr;
        if(isArr1Rvalue)
            result = const_cast<NDArray*>(&arr1);
        else if(isArr2Rvalue)
            result = const_cast<NDArray*>(&arr2);
        else
            result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()), false, arr1.getContext());

        NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
        NativeOpExecutioner::execPairwiseTransform(arr1.getContext(), sd::pairwise::Subtract, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(), result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({result}, {&arr1, &arr2});

        if(!isArr1Rvalue && !isArr2Rvalue) {
            NDArray res = std::move(*result);
            delete result;
            return std::move(res);
        }

        return std::move(*result);
    }

    return std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), std::forward<T2>(arr2));
}
template ND4J_EXPORT NDArray operator-<NDArray&, NDArray&, void>(NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator-<NDArray&, NDArray, void>(NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator-<NDArray,  NDArray&, void>(NDArray&& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator-<NDArray&, const NDArray&, void>(NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator-<const NDArray&, NDArray&, void>(const NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator-<const NDArray&, NDArray, void>(const NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator-<const NDArray&, const NDArray&, void>(const NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator-<NDArray, const NDArray&, void>(NDArray&& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator-<NDArray, NDArray, void>(NDArray&& arr1, NDArray&& arr2);

////////////////////////////////////////////////////////////////////////
// multiplication operator array*array
template <typename T1, typename T2, typename>
NDArray operator*(T1&& arr1, T2&& arr2) {

    if (arr1.isS() || arr2.isS())
        throw std::runtime_error("operator*(T&& arr1, T&& arr2): you can't use this method on String arrays!");
    if (!Environment::getInstance()->isExperimentalBuild() && arr1.dataType() != arr2.dataType() && (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
        throw sd::datatype_exception::build("operator*(T&& arr1, T&& arr2): Cannot multiply different types", arr1.dataType(), arr2.dataType());

    PointersManager pointersManager(arr1.getContext(), "operator*(T&& arr1, T&& arr2)");

    if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {

        const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
        const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

        NDArray* result = nullptr;
        if(isArr1Rvalue)
            result = const_cast<NDArray*>(&arr1);
        else if(isArr2Rvalue)
            result = const_cast<NDArray*>(&arr2);
        else
            result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()), false, arr1.getContext());

        NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
        NativeOpExecutioner::execPairwiseTransform(arr1.getContext(), sd::pairwise::Multiply, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(), result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({result}, {&arr1, &arr2});

        if(!isArr1Rvalue && !isArr2Rvalue) {
            NDArray res = std::move(*result);
            delete result;
            return std::move(res);
        }

        return std::move(*result);
    }

    return std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), std::forward<T2>(arr2));
}
template ND4J_EXPORT NDArray operator*<NDArray&, NDArray&, void>(NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator*<NDArray&, NDArray, void>(NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator*<NDArray,  NDArray&, void>(NDArray&& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator*<NDArray&, const NDArray&, void>(NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator*<const NDArray&, NDArray&, void>(const NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator*<const NDArray&, NDArray, void>(const NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator*<const NDArray&, const NDArray&, void>(const NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator*<NDArray, const NDArray&, void>(NDArray&& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator*<NDArray, NDArray, void>(NDArray&& arr1, NDArray&& arr2);

////////////////////////////////////////////////////////////////////////
// multiplication operator array*array
template <typename T1, typename T2, typename>
NDArray operator/(T1&& arr1, T2&& arr2) {

    if (arr1.isS() || arr2.isS())
        throw std::runtime_error("operator/(T&& arr1, T&& arr2): you can't use this method on String arrays!");
    if (!Environment::getInstance()->isExperimentalBuild() && arr1.dataType() != arr2.dataType() && (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
        throw sd::datatype_exception::build("operator/(T&& arr1, T&& arr2): Cannot multiply different types", arr1.dataType(), arr2.dataType());

    PointersManager pointersManager(arr1.getContext(), "operator/(T&& arr1, T&& arr2)");

    if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {

        const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
        const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

        NDArray* result = nullptr;
        if(isArr1Rvalue)
            result = const_cast<NDArray*>(&arr1);
        else if(isArr2Rvalue)
            result = const_cast<NDArray*>(&arr2);
        else
            result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()), false, arr1.getContext());

        NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
        NativeOpExecutioner::execPairwiseTransform(arr1.getContext(), sd::pairwise::Divide, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(), result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
        NDArray::registerSpecialUse({result}, {&arr1, &arr2});

        if(!isArr1Rvalue && !isArr2Rvalue) {
            NDArray res = std::move(*result);
            delete result;
            return std::move(res);
        }

        return std::move(*result);
    }

    return std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), std::forward<T2>(arr2));
}
template ND4J_EXPORT NDArray operator/<NDArray&, NDArray&, void>(NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator/<NDArray&, NDArray, void>(NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator/<NDArray,  NDArray&, void>(NDArray&& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator/<NDArray&, const NDArray&, void>(NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator/<const NDArray&, NDArray&, void>(const NDArray& arr1, NDArray& arr2);
template ND4J_EXPORT NDArray operator/<const NDArray&, NDArray, void>(const NDArray& arr1, NDArray&& arr2);
template ND4J_EXPORT NDArray operator/<const NDArray&, const NDArray&, void>(const NDArray& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator/<NDArray, const NDArray&, void>(NDArray&& arr1, const NDArray& arr2);
template ND4J_EXPORT NDArray operator/<NDArray, NDArray, void>(NDArray&& arr1, NDArray&& arr2);


/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif












//////////////////////////////////////////////////////////////////////////
// check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
// bool NDArray::hasOrthonormalBasis(const int arg) {
//     if (isS())
//         throw std::runtime_error("NDArray::hasOrthonormalBasis: you can't use this method on String array!");
//     if(rankOf() !=2 )
//         throw std::runtime_error("NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !");

//     if(arg!=0  && arg!=1)
//         throw std::runtime_error("NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !");

//     const double eps = 1e-5;
//     double dot = 0.f;

//     if(arg) {                   // check whether columns create orthogonal basis
//         for(int j=0; j<columns()-1; ++j)
//             for(int k=j+1; k<columns(); ++k) {
//                 for(int i=0; i<rows(); ++i)
//                     dot += e<double>(i,j)*e<double>(i,k);

//                 if(sd::math::nd4j_abs(dot) > eps )
//                     return false;

//                 dot = 0.f;
//             }

//             for(int j=0; j<columns(); ++j)  {   // check whether norm of column vector = 1
//                 for(int i=0; i<rows(); ++i)
//                     dot += e<double>(i,j)*e<double>(i,j);
//             if(dot != 0.f && sd::math::nd4j_abs(sd::math::nd4j_sqrt<double, double>(dot) - 1.f) > eps)
//                 return false;

//             dot = 0.f;
//         }
//     }
//     else {                      // check whether rows create orthogonal basis
//         for(int i=0; i<rows()-1; ++i)
//             for(int k=i+1; k<rows(); ++k) {
//                 for(int j=0; j<columns(); ++j)
//                     dot += e<double>(i,j)*e<double>(k,j);

//                 if(sd::math::nd4j_abs(dot) > eps )
//                     return false;

//                 dot = 0.;
//             }

//             for(int i=0; i<rows(); ++i) {       // check whether norm of row vector = 1
//                 for(int j=0; j<columns(); ++j)
//                     dot += e<double>(i,j)*e<double>(i,j);

//                 if(dot!= 0. && sd::math::nd4j_abs(sd::math::nd4j_sqrt<double, double>(dot) - 1.) > eps)
//                     return false;
//                 dot = 0.;
//             }
//         }
//     return true;
// }
