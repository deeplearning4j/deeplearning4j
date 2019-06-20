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

#ifndef NDARRAY_H
#define NDARRAY_H

#include <dll.h>
#include <initializer_list>
#include <functional>
#include <shape.h>
#include "NativeOpExecutioner.h"
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <graph/Intervals.h>
#include <array/DataType.h>
#include <stdint.h>
#include <array/ArrayOptions.h>
#include <array/ArrayType.h>
#include <array/ResultSet.h>
#include <helpers/ShapeBuilders.h>
#include <op_enums.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/BroadcastBoolOpsTuple.h>
#include <array/ExtraArguments.h>
#include <Status.h>
#include <ShapeDescriptor.h>
#include <helpers/ConstantShapeHelper.h>
#include <array/DataBuffer.h>


namespace nd4j {


    ND4J_EXPORT NDArray operator-(const float&, const NDArray&);
    ND4J_EXPORT NDArray operator-(const float16&, const NDArray&);
    ND4J_EXPORT NDArray operator-(const double&, const NDArray&);
    ND4J_EXPORT NDArray operator-(const int&, const NDArray&);

    ND4J_EXPORT NDArray operator+(const float&, const NDArray&);
    ND4J_EXPORT NDArray operator+(const float16&, const NDArray&);
    ND4J_EXPORT NDArray operator+(const double&, const NDArray&);
    ND4J_EXPORT NDArray operator+(const int&, const NDArray&);

    ND4J_EXPORT NDArray operator*(const float&, const NDArray&);
    ND4J_EXPORT NDArray operator*(const float16&, const NDArray&);
    ND4J_EXPORT NDArray operator*(const double&, const NDArray&);
    ND4J_EXPORT NDArray operator*(const int&, const NDArray&);

    ND4J_EXPORT NDArray operator/(const float&, const NDArray&);
    ND4J_EXPORT NDArray operator/(const float16&, const NDArray&);
    ND4J_EXPORT NDArray operator/(const double&, const NDArray&);
    ND4J_EXPORT NDArray operator/(const int&, const NDArray&);

    ND4J_EXPORT NDArray mmul(const NDArray&, const NDArray&);

    class ND4J_EXPORT NDArray {
    private:
        /**
         * This method applies given value to the buffer, wrt templates
         * @tparam T
         * @tparam Y
         * @param buffer
         * @param indices
         * @param value
         */
        template <typename T, typename Y>
        void templatedSet(void *buffer, const Nd4jLong *indices, const void *value);

        template <typename T, typename Y>
        void templatedSet(void *buffer, const Nd4jLong xOffset, const void *value);

        template <typename T>
        void templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value);

        template <typename T>
        void templatedAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const;

        template <typename X, typename Y>
        void templatedDoubleAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const;

        template <typename T, typename R>
        FORCEINLINE R templatedGet(void *buffer, const Nd4jLong index) const;
/*
        template <typename T, typename R>
        R templatedGetIndex(void *buffer, Nd4jLong *indices) const;
*/
        template <typename T>
        void* templatedPointerShift(const Nd4jLong offset) const;

        FORCEINLINE void copyBufferStatus(const NDArray& other) const;

    protected:

       /**
       *  if true then array doesn't own buffer and simply points to another's buffer
       */
        bool _isView = false;

        /**
        *  pointer on DataBuffer buffers in cpu/device memory
        */
        std::shared_ptr<DataBuffer> _buffer = std::make_shared<DataBuffer>();

        /**
        *  buffers offset, it is the same both for cpu and device buffers
        */
        Nd4jLong _offset = 0L;

        /**
        *  contains shape info:  matrix rank, numbers of elements per each dimension, dimensions strides, element-wise-stride, c-like or fortan-like order
        */
        Nd4jLong *_shapeInfo = nullptr;
        Nd4jLong *_shapeInfoD = nullptr;

        /**
        *  pointer on device launch context (with all data needed there).
        */
        nd4j::LaunchContext * _context = nd4j::LaunchContext::defaultContext();

        // indicates if array's buffer is within workspace
        bool _isAttached = false;

        /**
         * Field to store cached length
         */
        Nd4jLong _length = -1L;

        /**
        *  type of array elements
        */
        nd4j::DataType _dataType = FLOAT32;

        template<typename T>
        std::string toStringValue(T value);

    public:
        NDArray() = default;

        /**
        *  do not allocate memory, memory for array is passed from outside
        */
#ifndef __JAVACPP_HACK__
        NDArray(std::shared_ptr<DataBuffer> buffer, const ShapeDescriptor& descriptor, nd4j::LaunchContext* context = nd4j::LaunchContext::defaultContext(), const Nd4jLong offset = 0);

        NDArray(std::shared_ptr<DataBuffer> buffer, const char order, const std::vector<Nd4jLong> &shape, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

#endif

        /**
        *  do not allocate memory, memory for array is passed from outside
        */
        NDArray(void *buffer, Nd4jLong* shapeInfo, nd4j::LaunchContext * context = nd4j::LaunchContext::defaultContext(), const bool isBuffAlloc = false);

        /**
        *  do not allocate memory, memory for array is passed from outside
        *  we suppose the content of both (device and host) buffers is identical
        */
        NDArray(void *buffer, void *bufferD, Nd4jLong* shapeInfo, nd4j::LaunchContext * context = nd4j::LaunchContext::defaultContext(), const bool isBuffAlloc = false, const bool isBuffDAlloc = false);

        /**
        *  copy constructor
        */
        NDArray(const NDArray& other);

        /**
        *  move constructor
        */
        NDArray(NDArray&& other) noexcept;

        /**
        *  constructor, create array stored at given workspace
        */
        NDArray(nd4j::LaunchContext * context);


        /**
		*  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to zeros, if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently
        */
		NDArray(Nd4jLong* shapeInfo, const bool copyStrides = false, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

        /**
        *  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to be zeros, if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently
        *  set dtype as array type
        */
        NDArray(Nd4jLong* shapeInfo, const nd4j::DataType dtype, const bool copyStrides = false, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

        /**
        *  this constructor creates new array using shape information contained in vector argument
        */
        NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype = DOUBLE, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

        /**
        * This constructor creates new array with elements copied from data and using shape information stored in shape, elements from data will be casted to dtype
        */
        NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, nd4j::DataType dtype = DOUBLE, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

        /**
        *  this constructor creates new array using given buffer (without memory allocating) and shape information stored in shape
        */
        NDArray(void *buffer, const char order, const std::vector<Nd4jLong> &shape,  nd4j::DataType dtype, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext());

        /**
        *  this constructor creates new NDArray with shape matching "other" array,
        *  doesn't copy "other" elements into new array !!!
        */
        explicit NDArray(const NDArray* other, const bool copyStrides = false, nd4j::LaunchContext* context = nd4j::LaunchContext ::defaultContext());

        /**
        *  this constructor creates scalar(and set its value = 0) or empty array depending on bool argument isScalar
        */
        NDArray(nd4j::DataType dtype, nd4j::LaunchContext * context = nd4j::LaunchContext ::defaultContext(), const bool isScalar = true);

        /**
         * This method blocks until asynchronous operation finishes
         */
        void synchronize(const char* msg) const;

        /**
         * This method allows to set _isAttached flag
         * @param reallyAttached
         */
        void setAttached(bool reallyAttached);

        void tickWriteHost() const;
        void tickWriteDevice() const;
        void tickReadHost() const;
        void tickReadDevice() const;
        void tickBothActual() const;
        bool isActualOnHostSide() const;
        bool isActualOnDeviceSide() const;
        void makeBothBuffersActual() const;

        void syncToHost() const;
        void syncToDevice() const;
        void syncShape() const;

        /**
         * This method can be used on architectures that use special buffers
         * @param writeList
         * @param readList
         */
        static void registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList);
        static void prepareSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables = false);

        static void registerPrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList);
        static void preparePrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables = false);

        /**
         * This method returns buffer pointer offset by given number of elements, wrt own data type
         * @param offset
         * @return
         */
        void *bufferWithOffset(Nd4jLong offset) const;

        void* specialBufferWithOffset(Nd4jLong offset) const;
        /**
        *  copy assignment operator
        *  in particular, when _dataType != other._dataType and both shapes are the same, there will be allocation of new _buffer and _dataType acquires other._dataType
        */
        NDArray& operator=(const NDArray& other);

        /**
        *  move assignment operator
        */
        NDArray& operator=(NDArray&& other) noexcept;

        /**
        *  assignment operator, assigns the same scalar to all array elements
        */
        template <typename T>
        NDArray& operator=(const T scalar);


        /**
        *   operators for memory allocation and deletion
        */
        void* operator new(size_t i);
        void operator delete(void* p);


        void setContext(nd4j::LaunchContext * context);

        /**
        *  create a new array by replicating current array by repeats times along given dimension
        *  dimension - dimension along which to repeat elements
        *  repeats - number of repetitions
        */
        NDArray* repeat(int dimension, const std::vector<Nd4jLong>& repeats) const;

        /**
         * This method fills this array with zeros
         */
        void nullify();

        /**
         * This method returns quantized copy of given array
         *
         * @param array
         * @return
         */
        static NDArray quantize(NDArray &array);

        /**
         * This method returns quantized copy of given array
         *
         * @param array
         * @return
         */
        static NDArray* quantize(NDArray *array);

        /**
        *  fill target array by repeating current array
        *  dimension - dimension along which to repeat elements
        */
        void repeat(int dimension, NDArray& target) const;

        /**
        *  creates array which points on certain sub-range of this array, sub-range is defined by given indices
        */
        NDArray* subarray(IndicesList& indices) const;
        NDArray* subarray(const std::initializer_list<NDIndex*>& idx) const;
        NDArray* subarray(const Intervals& idx) const;

        /**
        *  cast array elements to given dtype
        */
        template <typename T>
        NDArray* cast();

        NDArray* cast(DataType dtype) const;

        void cast(NDArray* target, DataType dtype);

        /**
        *   returns _context
        */
        nd4j::LaunchContext * getContext() const {
            return _context;
        }

#ifndef __JAVACPP_HACK__
        FORCEINLINE std::shared_ptr<DataBuffer> getDataBuffer() const;
        FORCEINLINE std::shared_ptr<DataBuffer> dataBuffer();
#endif

        /**
        *   returns host buffer
        */
        FORCEINLINE void* getBuffer() const;
        FORCEINLINE void* buffer();


        /**
        *   returns buffer offset (offset is the same for host and device buffers)
        */
        FORCEINLINE Nd4jLong getBufferOffset() const;
        FORCEINLINE Nd4jLong bufferOffset();

        /**
        *  if _bufferD==nullptr return _buffer, else return _bufferD
        */
        void* specialBuffer();
        void* getSpecialBuffer() const;

        /**
        *   returns device buffer if compilation is for cuda case, otherwise returns host buffer
        */
        void* getPlatformBuffer() const;
        void* platformBuffer();



        template <typename T>
        T* bufferAsT() const;

        /**
        *   returns _shapeInfo
        */
        FORCEINLINE Nd4jLong* shapeInfo();
        FORCEINLINE Nd4jLong* getShapeInfo() const;


        /**
         * Returns True if it's legally empty NDArray, or false otherwise
         * @return
         */
        FORCEINLINE bool isEmpty() const;

        /**
        *  if _shapeInfoD==nullptr return _shapeInfo, else return _shapeInfoD
        */
        FORCEINLINE Nd4jLong* specialShapeInfo();
        FORCEINLINE Nd4jLong* getSpecialShapeInfo() const;


        Nd4jLong* platformShapeInfo();
        Nd4jLong* getPlatformShapeInfo() const;

        /**
        *  permutes (in-place) the dimensions in array according to "dimensions" array
        */
        bool permutei(const std::initializer_list<int>& dimensions);
        bool permutei(const std::vector<int>& dimensions);
        bool permutei(const int* dimensions, const int rank);

        bool permutei(const std::initializer_list<Nd4jLong>& dimensions);
        bool permutei(const std::vector<Nd4jLong>& dimensions);
        bool permutei(const Nd4jLong* dimensions, const int rank);

        bool isFinite();
        bool hasNaNs();
        bool hasInfs();

        void copyBuffersContinuouslyFrom(const NDArray& other, size_t sizeToCopyInBytes = 0, Nd4jLong offsetThis = 0, Nd4jLong offsetOther = 0);

        /**
        *  permutes the dimensions in array according to "dimensions" array, new array points on _buffer of this array
        */
		NDArray permute(const std::initializer_list<int>& dimensions) const;
        NDArray permute(const std::vector<int>& dimensions) const;
        NDArray permute(const int* dimensions, const int rank) const;

        void permute(const int* dimensions, const int rank, NDArray& target) const;
        void permute(const std::vector<int>& dimensions, NDArray& target) const;

        NDArray permute(const std::initializer_list<Nd4jLong>& dimensions) const;
        NDArray permute(const std::vector<Nd4jLong>& dimensions) const;
        NDArray permute(const Nd4jLong* dimensions, const int rank) const;

        void permute(const Nd4jLong* dimensions, const int rank, NDArray& target) const;
        void permute(const std::vector<Nd4jLong>& dimensions, NDArray& target) const;

        /**
         * This method streamlines given view or permuted array, and reallocates buffer
         */
        void streamline(char order = 'a');

        /**
        *  prints information about array shape
        *  msg - message to print out
        */
        void printShapeInfo(const char * msg = nullptr) const;

        /**
        *  prints buffer elements
        *  msg - message to print out
        *  limit - number of array elements to print out
        *  sync - if true check whether host buffer is actual, if it is not then make it so
        */
        void printBuffer(const char* msg = nullptr, Nd4jLong limit = -1, const bool sync = true) const;

        /**
        *  prints _buffer (if host = true) or _bufferD (if host = false) as it is, that is in current state without checking buffer status
        */
        template<typename T>
        void printCurrentBuffer(const bool host = true, const char* msg = nullptr, const int precision = 1) const;

        /**
        *  prints buffer elements, takes into account offset between elements (element-wise-stride)
        *  msg - message to print out
        *  limit - number of array elements to print out
        */
        void printIndexedBuffer(const char* msg = nullptr, Nd4jLong limit = -1) const;

        std::string asIndexedString(Nd4jLong limit = -1);
        std::string asString(Nd4jLong limit = -1);

        /**
        *  this method assigns values of given array to this one
        */
        void assign(const NDArray* other);

        /**
        *  this method assigns values of given array to this one
        */
        void assign(const NDArray& other);

        /**
        *  this method assigns given value to all elements in array
        */
        void assign(const double value);
        void assign(const float value);
        void assign(const float16 value);
        void assign(const bfloat16& value);
        void assign(const Nd4jLong value);
        void assign(const int value);
        void assign(const int16_t value);
        void assign(const uint8_t value);
        void assign(const uint16_t value);
        void assign(const uint32_t value);
        void assign(const uint64_t value);
        void assign(const int8_t value);
        void assign(const bool value);

        /**
        *  returns new copy of this array, optionally in different order
        */
        NDArray *dup(const char newOrder = 'a');

        /**
        *  returns sum of all elements of array
        */
        NDArray sumNumber() const;

        /**
        *  returns mean number of array
        */
        NDArray meanNumber() const;

#ifndef __JAVACPP_HACK__

        /**
         * This method explicitly enforces new shape for this NDArray, old shape/stride information is lost
         */
        void enforce(const std::initializer_list<Nd4jLong> &dimensions, char order = 'a');
        void enforce(std::vector<Nd4jLong> &dimensions, char order = 'a');


		/**
        *  method reduces array by excluding its shapes along dimensions present in given dimensions vector, result is stored in new array to be returned
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        */

        NDArray* reduceAlongDimension(nd4j::reduce::FloatOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray* reduceAlongDimension(nd4j::reduce::FloatOps op, const std::initializer_list<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray  reduceAlongDims(nd4j::reduce::FloatOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;

        NDArray* reduceAlongDimension(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray* reduceAlongDimension(nd4j::reduce::SameOps op, const std::initializer_list<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray  reduceAlongDims(nd4j::reduce::SameOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;

        NDArray* reduceAlongDimension(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray* reduceAlongDimension(nd4j::reduce::BoolOps op, const std::initializer_list<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray  reduceAlongDims(nd4j::reduce::BoolOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;

        NDArray* reduceAlongDimension(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray* reduceAlongDimension(nd4j::reduce::LongOps op, const std::initializer_list<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        NDArray  reduceAlongDims(nd4j::reduce::LongOps op, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;

        /**
        *  method reduces array by excluding its shapes along dimensions present in given dimensions vector
        *  target - where to save result of reducing
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        *  extras - extra parameters
        */
        void reduceAlongDimension(nd4j::reduce::FloatOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false, const bool checkTargetShape = true) const;
        void reduceAlongDimension(nd4j::reduce::SameOps op, NDArray* target,  const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false, const bool checkTargetShape = true) const;
        void reduceAlongDimension(nd4j::reduce::BoolOps op, NDArray* target,  const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false, const bool checkTargetShape = true) const;
        void reduceAlongDimension(nd4j::reduce::LongOps op, NDArray* target,  const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false, const bool checkTargetShape = true) const;

        /**
        *  return variance of array elements set
        *  biasCorrected -  if true bias correction will be applied
        */
        NDArray varianceNumber(nd4j::variance::Ops op, bool biasCorrected = true);

        /**
        *  apply scalar operation to array
        *  extraParams - extra parameters for operation
        */
        NDArray reduceNumber(nd4j::reduce::FloatOps ops, void *extraParams = nullptr) const;
        NDArray reduceNumber(nd4j::reduce::SameOps  ops, void *extraParams = nullptr) const;
        NDArray reduceNumber(nd4j::reduce::BoolOps  ops, void *extraParams = nullptr) const;
        NDArray reduceNumber(nd4j::reduce::LongOps  ops, void *extraParams = nullptr) const;

        void reduceNumber(nd4j::reduce::FloatOps ops, NDArray& target, void *extraParams = nullptr) const;
        void reduceNumber(nd4j::reduce::SameOps  ops, NDArray& target, void *extraParams = nullptr) const;
        void reduceNumber(nd4j::reduce::BoolOps  ops, NDArray& target, void *extraParams = nullptr) const;
        void reduceNumber(nd4j::reduce::LongOps  ops, NDArray& target, void *extraParams = nullptr) const;

        /**
        *  returns element index which corresponds to some condition imposed by operation
        *  extraParams - extra parameters for operation
        */
        NDArray indexReduceNumber(nd4j::indexreduce::Ops op, ExtraArguments *extraParams = nullptr);

        /**
        *  returns index of max element in a given array (optionally: along given dimension(s))
        *  dimensions - optional vector with dimensions
        */
        Nd4jLong argMax(std::initializer_list<int> dimensions = {});

        // FIXME: remove this method eventually
        void makeBothActual() const        { syncToDevice(); syncToHost(); }


        void applyTransform(nd4j::transform::FloatOps op, NDArray *target = nullptr, ExtraArguments *extraParams = nullptr);
        void applyTransform(nd4j::transform::SameOps op, NDArray *target = nullptr, ExtraArguments *extraParams = nullptr);
        void applyTransform(nd4j::transform::AnyOps op, NDArray *target = nullptr, ExtraArguments *extraParams = nullptr);
        void applyTransform(nd4j::transform::BoolOps op, NDArray *target = nullptr, ExtraArguments *extraParams = nullptr);
        void applyTransform(nd4j::transform::StrictOps op, NDArray *target = nullptr, ExtraArguments *extraParams = nullptr);

        /**
        *  apply OpName transformation to this array and store result in new array to be returned
        *  extraParams - extra parameters for operation
        */
        NDArray transform(nd4j::transform::FloatOps op, void *extraParams = nullptr) const;
        NDArray transform(nd4j::transform::SameOps op, void *extraParams = nullptr) const;
        NDArray transform(nd4j::transform::BoolOps op, void *extraParams = nullptr) const;
        NDArray transform(nd4j::transform::StrictOps op, void *extraParams = nullptr) const;

        /**
        *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in this array
        *  other - second array necessary for pairwise operation
        *  extraParams - extra parameters for operation
        */
        void applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray& other, ExtraArguments *extraParams = nullptr);

        /**
        *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in target array
        *  other - second array necessary for pairwise operation
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        void applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray *other, NDArray *target, ExtraArguments *extraParams = nullptr) const;

        void applyPairwiseTransform(nd4j::pairwise::BoolOps op, const NDArray *other, NDArray *target, ExtraArguments *extraParams = nullptr) const;

        /**
        *  apply operation which requires broadcasting, broadcast a smaller array (tad) along  bigger one (this)
        *  tad - array to broadcast
        *  dimensions -  dimensions array to broadcast along
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        void applyBroadcast(nd4j::broadcast::Ops op, const std::initializer_list<int> dimensions, const NDArray* tad, NDArray* target = nullptr, ExtraArguments* extraArgs = nullptr);

        void applyBroadcast(nd4j::broadcast::Ops op, const std::vector<int> &dimensions, const NDArray *tad, NDArray *target = nullptr, ExtraArguments *extraArgs = nullptr);

        void applyBroadcast(nd4j::broadcast::BoolOps op, const std::vector<int> &dimensions, const NDArray *tad, NDArray *target = nullptr, ExtraArguments *extraArgs = nullptr);

        /**
        *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the possibility of broadcasting
        *  other - input array
        *  extraParams - extra parameters for operation
        */
        NDArray applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs = nullptr) const;

        NDArray* applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, ExtraArguments *extraArgs = nullptr) const;

        /**
        *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the possibility of broadcasting
        *  other - input array
        *  target - where to store result
        *  checkTargetShape - if true check whether target shape is suitable for broadcasting
        *  extraParams - extra parameters for operation
        */
        void applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape = true, ExtraArguments *extraArgs = nullptr) const;

        void applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape = true, ExtraArguments *extraArgs = nullptr) const;

        /**
        *  apply a scalar operation to an array
        *  scalar - input scalar
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        template <typename T>
        void applyScalar(nd4j::scalar::Ops op, const T scalar, NDArray* target = nullptr, ExtraArguments *extraParams = nullptr);

        template <typename T>
        void applyScalar(nd4j::scalar::BoolOps op, const T scalar, NDArray* target, ExtraArguments *extraParams = nullptr) const;

        /**
        *  apply a scalar operation to an array
        *  scalar - input array which is simple scalar
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        void applyScalarArr(nd4j::scalar::Ops op, const NDArray* scalar, NDArray* target = nullptr, ExtraArguments *extraParams = nullptr);

        void applyScalarArr(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray* target, ExtraArguments *extraParams = nullptr) const;


#if defined(__CUDABLAS__) //&& defined(BUILD_TESTS)
        template <typename Lambda>
        FORCEINLINE void applyLambda(Lambda func, NDArray* target = nullptr);

        template <typename Lambda>
        FORCEINLINE void applyPairwiseLambda(const NDArray* other, Lambda func, NDArray* target = nullptr);

        template <typename Lambda>
        FORCEINLINE void applyIndexedLambda(Lambda func, NDArray* target = nullptr);

        template <typename Lambda>
        FORCEINLINE void applyIndexedPairwiseLambda(NDArray* other, Lambda func, NDArray* target = nullptr);

        template <typename Lambda>
        FORCEINLINE void applyTriplewiseLambda(NDArray* second, NDArray *third, Lambda func, NDArray* target = nullptr);
#else

        /**
        *  apply operation "func" to an array
        *  func - what operation to apply
        *  target - where to store result
        */
        template <typename T>
        void applyLambda(const std::function<T(T)>& func, NDArray* target = nullptr);

        /**
        *  apply pairwise operation "func" to an array
        *  other - input array
        *  func - what pairwise operation to apply
        *  target - where to store result
        */
        template <typename T>
        void applyPairwiseLambda(const NDArray* other, const std::function<T(T, T)>& func, NDArray* target = nullptr);

        template <typename T>
        void applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target = nullptr);

        template <typename T>
        void applyIndexedPairwiseLambda(NDArray* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray* target = nullptr);

        template <typename T>
        void applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<T(T, T, T)>& func, NDArray* target = nullptr);
#endif

        /**
        *  reduces dimensions in this array relying on index operation OpName
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        NDArray* applyIndexReduce(nd4j::indexreduce::Ops op, const std::vector<int>& dimensions, const ExtraArguments *extraParams = nullptr) const;

        /**
        *  reduces dimensions in array relying on index operation OpName
        *  target - where to store result
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        void applyIndexReduce(nd4j::indexreduce::Ops op, NDArray* target, const std::vector<int>& dimensions, const ExtraArguments *extraParams = nullptr) const;

        /**
        *  apply reduce3 operation OpName to this and other array, return result in new output array
        *  other - input array
        *  extraArgs - extra parameters for operation
        */
        NDArray* applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const ExtraArguments* extraParams = nullptr) const;

        /**
        *  apply reduce3 operation OpName to this and other array, return result in new output array
        *  other - input array
        *  dimensions - vector of dimensions to reduce along (tads not axis)
        *  extraArgs - extra parameters for operation
        */
        NDArray* applyAllReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const ExtraArguments* extraParams = nullptr) const;

        /**
        *  apply reduce3 (exec) operation OpName to this and other array, return result in new output array
        *  other - input array
        *  dimensions - vector of dimensions to reduce along (same as reduceAlongDimension)
        *  extraArgs - extra parameters for operation
        */
        NDArray* applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const ExtraArguments* extraParams = nullptr) const;


        /**
        *  returns variance along given dimensions
        *  biasCorrected -  if true bias correction will be applied
        *  dimensions - vector of dimensions to calculate variance along
        */
        NDArray* varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const;
        NDArray* varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const;

        NDArray varianceAlongDims(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const;
        NDArray varianceAlongDims(nd4j::variance::Ops op, const bool biasCorrected, const std::initializer_list<int>& dimensions) const;

        void varianceAlongDimension(nd4j::variance::Ops op, NDArray* target, const bool biasCorrected, const std::vector<int>& dimensions) const;

        void varianceAlongDimension(nd4j::variance::Ops op, NDArray* target, const bool biasCorrected, const std::initializer_list<int>& dimensions) const;

#endif

        /**
        *   apply transpose operation to the copy of this array, that is this array remains unaffected
        */
        NDArray transpose() const;

        /**
        *  perform transpose operation and store result in target, this array remains unaffected
        *  target - where to store result
        */
        void transpose(NDArray& target) const;

        /**
        *  apply in-place transpose operation to this array, so this array becomes transposed
        */
        void transposei();

        /**
        *  return array pointing on certain range of this array
        *  index - the number of array to be returned among set of possible arrays
        *  dimensions - array of dimensions to point on
        */
        NDArray* tensorAlongDimension(Nd4jLong index, const std::initializer_list<int>& dimensions) const;
        NDArray* tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const;

        /**
        *  returns the number of arrays pointing on specified dimension(s)
        *  dimensions - array of dimensions to point on
        */
        Nd4jLong tensorsAlongDimension(const std::initializer_list<int> dimensions) const ;
        Nd4jLong tensorsAlongDimension(const std::vector<int>& dimensions) const ;

        /**
        *  returns true if elements of two arrays are equal to within given epsilon value
        *  other - input array to compare
        *  eps - epsilon, this value defines the precision of elements comparison
        */
        bool equalsTo(const NDArray *other, double eps = 1e-5) const;
        bool equalsTo(const NDArray &other, double eps = 1e-5) const;

        /**
        *  add given row vector to all rows of this array
        *  row - row vector to add
        */
        void addiRowVector(const NDArray *row);

        /**
        *  add given row vector to all rows of this array, store result in target
        *  row - row vector to add
        *  target - where to store result
        */
        void addRowVector(const NDArray *row, NDArray* target) const;

        /**
        *  subtract given row vector from all rows of this array, store result in target
        *  row - row vector to subtract
        *  target - where to store result
        */
        void subRowVector(const NDArray *row, NDArray* target) const;

        /**
        *  multiply all rows of this array on given row vector, store result in target
        *  row - row vector to multiply on
        *  target - where to store result
        */
        void mulRowVector(const NDArray *row, NDArray* target) const;

        /**
        *  divide all rows of this array on given row vector, store result in target
        *  row - row vector to divide on
        *  target - where to store result
        */
        void divRowVector(const NDArray *row, NDArray* target) const;

        /**
        *  add given column vector to all columns of this array, store result in target
        *  column - column vector to add
        *  target - where to store result
        */
        void addColumnVector(const NDArray *column, NDArray* target) const;

        /**
        *  add given column vector to all columns of this array, this array becomes affected (in-place operation)
        *  column - column vector to add
        */
		void addiColumnVector(const NDArray *column);

        /**
        *  multiply all columns of this array on given column vector, this array becomes affected (in-place operation)
        *  column - column vector to multiply on
        */
		void muliColumnVector(const NDArray *column);

        /**
        *  returns number of bytes used by _buffer & _shapeInfo
        */
        FORCEINLINE Nd4jLong memoryFootprint();

        /**
        *  these methods suited for FlatBuffers use
        */
        template <typename T>
        std::vector<T> getBufferAsVector();
        std::vector<Nd4jLong> getShapeAsVector() const;
        std::vector<Nd4jLong> getShapeInfoAsVector();
        std::vector<int64_t> getShapeInfoAsFlatVector();
        std::vector<int64_t> getShapeAsFlatVector();

        /**
        *  set new order and shape in case of suitable array length (in-place operation)
        *  order - order to set
        *  shape - shape to set
        *
        *  if there was permute applied before or there are weird strides, then new buffer is allocated for array
        */
		bool reshapei(const char order, const std::initializer_list<Nd4jLong>& shape);
		bool reshapei(const char order, const std::vector<Nd4jLong>& shape);

        bool reshapei(const std::initializer_list<Nd4jLong>& shape);
		bool reshapei(const std::vector<Nd4jLong>& shape);

        /**
        *  creates new array with corresponding order and shape, new array will point on _buffer of this array
        *  order - order to set
        *  shape - shape to set
        *
        * if permute have been applied before or there are weird strides, then new buffer is allocated for new array
        */
		NDArray reshape(const char order, const std::vector<Nd4jLong>& shape) const;

        /**
        *  calculate strides and set given order
        *  order - order to set
        */
		void updateStrides(const char order);

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        */
		void tilei(const std::vector<Nd4jLong>& repeats);

        /**
        *  returns new array which is created by repeating of this array the number of times given by reps
        *  repeats - contains numbers of repetitions
        */
		NDArray tile(const std::vector<Nd4jLong>& repeats) const;

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        *  target - where to store result
        */
        void tile(const std::vector<Nd4jLong>& repeats, NDArray& target) const;

        /**
        *  change an array by repeating it the number of times to acquire the new shape which is the same as target shape
        *  target - where to store result
        */
        void tile(NDArray& target) const;

        /**
        *  returns an array which is result of broadcasting of this and other arrays
        *  other - input array
        */
		NDArray* broadcast(const NDArray& other);

        /**
        *  check whether array is identity matrix
        */
		bool isIdentityMatrix();

        /**
        *  check whether array is unitary matrix
        */
		bool isUnitary();


        /**
        *  operator returns subarray with buffer pointing at this->_buffer with offset defined by given intervals
        *  idx - intervals of indexes which define the subarrays to point on, idx has form {dim0Start,dim0End,  dim1Start,dim1End, ....} and length (2 * this->rankOf())
        *        when (dimStart == dimEnd) then whole range will be used for current dimension
        *  keepUnitiesInShape - if false then eliminate unities from resulting array shape, for example {1,a,1,b} -> {a,b}
        *  isStrided - if true then idx has length (3 * this->rankOf()) and contains additional stride numbers which correspond to stride between dimStart and dimEnd,
        *              so structure of idx is like {dim0Start,dim0End,dim0Stride,    dim1Start,dim1End,dim1Stride, ....}
        */
        NDArray operator()(const std::vector<Nd4jLong>& idx, const bool keepUnitiesInShape = false, const bool isStrided = false)  const;

        /**
        *  evaluates subarray with buffer pointing at this->_buffer and offset defined by given sequential index subArrIdx and dimensions in dimsToExclude
        *  subArrIdx - index of current sub-array
        *  dimsToExclude - MUST BE SORTED, dimensions to evaluate sub-array along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays with shape [3,5], and subArrIdx must be in range [0,7]
        *                  if dimsToExclude is empty then idxRanges containing all zeros (means whole array) will be returned.
        *  keepUnitiesInShape - if false then eliminate unities from resulting array shape, for example {1,a,1,b} -> {a,b}
        */
        NDArray operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape = false)  const;

        /**
        * processes whole set of sub-arrays
        * evaluates shapeInfo of sub-arrays (all sub-arrays have the same shapeInfo) and their buffer offsets (each sub-array has its own unique offset from original this-buffer)
        * dimsToExclude - MUST BE SORTED, dimensions to evaluate sub-array along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays with shape [3,5]
        *                 if dimsToExclude.size() = array rank it means sub-array is whole array and copy of original_shapeInfo will be returned and one zero offset
        * subArrShapeInfo    - output argument, contains shapeInfo common for all sub-arrays
        * subArrOffsets      - output argument, contains successive sub-arrays offsets from original this-buffer
        * keepUnitiesInShape - if false then eliminate unities from sub-array shapeInfo, for example {1,a,1,b} -> {a,b}
        */
        void getSubArrShapeAndOffsets(const std::vector<int>& dimsToExclude, Nd4jLong* &subArrShapeInfo, Nd4jLong* &subArrOffsets, bool keepUnitiesInShape = false) const;

        /**
        *  addition operator: array + other
        *  other - input array to add
        */
        NDArray operator+(const NDArray& other) const;

        /**
        *  addition operator: array + scalar
        *  scalar - input scalar to add
        */
        template <typename T>
        NDArray operator+(const T& scalar) const;

        /**
        *  friend functions which implement addition operator: scalar + array
        *  scalar - input scalar to add
        */
        //template <typename T>
        //friend NDArray nd4j::operator+(const T scalar, const NDArray& arr);


        /**
        *  addition unary operator array += other
        *  other - input array to add
        */
        void operator+=(const NDArray& other);

        /**
        *  subtraction unary operator array -= other
        *  other - input array to add
        */
        void operator-=(const NDArray& other);

        template <typename T>
        void operator+=(const T other);

        template <typename T>
        void operator-=(const T other);

        /**
        *  subtraction operator: array - other
        *  other - input array to subtract
        */
        NDArray operator-(const NDArray& other) const;

        /**
        *  subtraction operator: array - scalar
        *  scalar - input scalar to subtract
        */
        template <typename T>
        NDArray operator-(const T& scalar) const;

        /**
        *  negative operator, it changes sign of all array elements on opposite
        */
        NDArray operator-() const;

        /**
        *  friend functions which implement subtraction operator: scalar - array
        *  scalar - input scalar to subtract
        */
        //friend NDArray nd4j::operator-(const float scalar, const NDArray& arr);

        /**
        *  pairwise multiplication operator: array * other
        *  other - input array to multiply on
        */
        NDArray operator*(const NDArray& other) const;

        /**
        *  multiplication operator: array * scalar
        *  scalar - input scalar to multiply on
        */
        template <typename T>
        NDArray operator*(const T& scalar) const;

        /**
        *  pairwise multiplication unary operator array *= other
        *  other - input array to multiply on
        */
        void operator*=(const NDArray& other);

        /**
        *  multiplication unary operator array *= scalar
        *  scalar - input scalar to multiply on
        */
        template <typename T>
        void operator*=(const T scalar);

        /**
        *  pairwise division operator: array / other
        *  other - input array to divide on
        */
        NDArray operator/(const NDArray& other) const;

        /**
        *  division operator: array / scalar
        *  scalar - input scalar to divide each array element on
        */
        template <typename T>
        NDArray operator/(const T& scalar) const;

        /**
        *  pairwise division unary operator: array /= other
        *  other - input array to divide on
        */
        void operator/=(const NDArray& other);

        /**
        *  division unary operator: array /= scalar
        *  scalar - input scalar to divide on
        */
        template <typename T>
        void operator/=(const T scalar);

        /**
        *  friend function which implements mathematical multiplication of two arrays
        *  left - input array
        *  right - input array
        */
        friend NDArray mmul(const NDArray& left, const NDArray& right);

        /**
        *  return vector containing _buffer as flat binary array
        */
        std::vector<int8_t> asByteVector();

        /**
        *  makes array to be identity matrix (not necessarily square), that is set all diagonal elements = 1, rest = 0
        */
        void setIdentity();

        /**
        *  swaps the contents of tow arrays,
        *  PLEASE NOTE: method doesn't take into account the shapes of arrays, shapes may be different except one condition: arrays lengths must be the same
        */
        void swapUnsafe(NDArray& other);

        /**
        *  return vector with buffer which points on corresponding diagonal elements of array
        *  type - means of vector to be returned: column ('c') or row ('r')
        */
        NDArray* diagonal(const char type ) const;

        /**
        * fill target matrix with given value in one or two directions from main diagonal:
        *   - down from main diagonal starting at subdiagonal number "lower" if direction = 'd' (down) or 'b' (both)
        *   - up from main diagonal starting at superdiagonal number "upper"if direction = 'u' (up) or 'b' (both)
        * direction - in what direction to fill matrix. There are 3 possible directions:
        *   'u' - fill up, mathematically this corresponds to lower triangular matrix, parameter "lower" is not taken into account
        *   'l' - fill down, mathematically this corresponds to upper triangular matrix, parameter "upper" is not taken into account
        *   'b' - fill in both directions, both parameters "lower" and "upper" are taken into account
        * rest of target elements are equal to this array elements
        * target and this array should have same shapes, except when this_rank = 1 (in that case should be target_rank = 2)
        */
        template <typename T>
        void fillAsTriangular(const float value, int lower, int upper, const char direction = 'b', NDArray* target = nullptr);

		/**
        *  change an array by repeating it the number of times in order to acquire new shape equal to the input shape
        *
        *  shape  - contains new shape to broadcast array to
        *  target - optional argument, if target != nullptr the resulting array will be placed in target, in opposite case tile operation is done in place
        */
        NDArray tileToShape(const Nd4jLong* shapeInfo);
        void tileToShape(const std::vector<Nd4jLong>& shape, NDArray* target = nullptr);
#ifndef __JAVACPP_HACK__
        void tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray* target = nullptr);
#endif

        template <typename N>
        NDArray* asT() const;

        NDArray* asT(DataType dtype) const;


        void linspace(const double start);

        void linspace(const double start, const double step);

        /**
        *  calculates the trace of an array, that is sum of elements on main diagonal = sum array[i, i, i, ...]
        */
        double getTrace() const;

        ResultSet* multipleTensorsAlongDimension(const std::vector<int>& indices, const std::vector<int>& dimensions) const;

        ResultSet* allTensorsAlongDimension(const std::initializer_list<int>& dimensions) const;

        ResultSet* allTensorsAlongDimension(const std::vector<int>& dimensions) const;

        //ResultSet  allTensorsAlongDims(const std::vector<int>& dimensions) const;

        ResultSet* allExamples()const ;

        /**
        *  set _shapeInfo
        */
        void setShapeInfo(const Nd4jLong *shapeInfo);
        void setShapeInfo(const Nd4jLong *shapeInfo, const nd4j::DataType dtype);
        void setShapeInfo(const ShapeDescriptor& descriptor);
        void setShapeInfo(const ConstantDataBuffer& shapeBuffer);

        /**
        *  returns absolute offset which corresponds to given sequential index
        */
        Nd4jLong getOffset(const Nd4jLong i) const;

        /**
        *  returns reference on array element with given index
        */
        template<typename T>
        FORCEINLINE T& t(const Nd4jLong index);

        template<typename T>
        FORCEINLINE T& t(const Nd4jLong i, const Nd4jLong j);

        /**
        *  returns array element with given index
        *  i - element index in array
        */
        template<typename T>
        FORCEINLINE T t(const Nd4jLong i) const;

        template<typename T>
        FORCEINLINE T t(const Nd4jLong i, const Nd4jLong j) const;


        /**
        *  default destructor
        */
        ~NDArray() noexcept = default;

        /**
        *  set _shapeInfo
        */
        FORCEINLINE void setShapeInfo(Nd4jLong *shapeInfo);
        FORCEINLINE void setShapeInfo(Nd4jLong *shapeInfo, const nd4j::DataType dtype);

        /**
        *  returns the value of "dim" dimension
        */
        Nd4jLong sizeAt(const int dim) const;

        /**
        *  returns order of array
        */
        FORCEINLINE char ordering() const;

        /**
        *  return _isView
        */
        FORCEINLINE bool isView() const;

        /**
        *  returns shape portion of shapeInfo
        */
        FORCEINLINE Nd4jLong* shapeOf() const;

        /**
        *  returns strides portion of shapeInfo
        */
        FORCEINLINE Nd4jLong* stridesOf() const;

        /**
        *  returns rank of array
        */
        FORCEINLINE int rankOf() const;

        /**
        *  returns length of array
        */
        FORCEINLINE Nd4jLong lengthOf() const;

        /**
        *  returns number of rows in array
        */
        FORCEINLINE Nd4jLong rows() const;

        /**
        *  returns number of columns in array
        */
        FORCEINLINE Nd4jLong columns() const;

        /**
        *  returns size of array elements type
        */
        FORCEINLINE size_t sizeOfT() const;

        /**
        *  returns element-wise-stride
        */
        FORCEINLINE Nd4jLong ews() const;

        // returns true if arrays have same shape
        FORCEINLINE bool isSameShape(const NDArray *other) const;
        FORCEINLINE bool isSameShape(const NDArray &other) const;
        FORCEINLINE bool isSameShape(const std::initializer_list<Nd4jLong>& shape) const;
        FORCEINLINE bool isSameShape(const std::vector<Nd4jLong>& shape) const;
        FORCEINLINE bool areSameShapeAndType(const NDArray& other) const;

        /**
        *  returns true if these two NDArrays have same rank, dimensions, strides, ews and order
        */
        FORCEINLINE bool isSameShapeStrict(const NDArray *other) const;

        /**
        *  returns true if buffer && shapeInfo were defined (non nullptr)
        */
        FORCEINLINE bool nonNull() const;

        template <typename T>
        T r(const Nd4jLong i) const;

        /**
        *  returns array element with given index from linear buffer
        *  i - element index in array
        */
        template <typename T>
        T e(const Nd4jLong i) const;

        /**
        *  returns element with given indexes from 2D array
        *  i - number of row
        *  j - number of column
        */
        template <typename T>
        T e(const Nd4jLong i, const Nd4jLong j) const;

        /**
        *  returns element with given indexes from 3D array
        *  i - height
        *  j - width
        *  k - depth
        */
        template <typename T>
        T e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const;

        /**
        *  returns element with given indexes from DD array
        */
        template <typename T>
        T e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l) const;

        /**
        *  returns array-scalar containing element of this array with given index
        *  i - element index in array
        */
        NDArray e(const Nd4jLong i) const;

        /**
        *  assigns given scalar to array element by given index, regards array buffer as linear
        *  i - element index in array
        *  value - scalar value to assign
        */
        template <typename T>
        void p(const Nd4jLong i, const T value);

        void p(const Nd4jLong i, const NDArray& value);

        /**
        *  assigns given scalar to 2D array element by given indexes
        *  i - number of row
        *  j - number of row
        *  value - scalar value to assign
        */
        template <typename T>
        void p(const Nd4jLong i, const Nd4jLong j, const T value);

        /**
        *  assigns given scalar to 3D array element by given indexes
        *  i - height
        *  j - width
        *  k - depth
        *  value - scalar value to assign
        */
        template <typename T>
        void p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value);

        template <typename T>
        void p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value);


        template <typename T>
        void pIdx(const Nd4jLong* indices, const T value);

        /**
        *  creates array which points on certain sub-range of this array, sub-range is defined by given indices
        */
        NDArray* subarray(IndicesList& indices, std::vector<Nd4jLong>& strides) const;

        /**
        *  returns true if array is 2D
        */
        FORCEINLINE bool isMatrix() const;

        /**
        *  returns true if array is vector
        */
        FORCEINLINE bool isVector() const;

        /**
        *  returns true if array is column vector
        */
        FORCEINLINE bool isColumnVector() const;

        /**
        *  returns true if array is row vector
        */
        FORCEINLINE bool isRowVector() const;

        /**
        *  returns true if all dimensions of array except one are unities, for example: [1,1,n,1], [n,1,1], [n], ...
        *  posOfNonUnityDim - one dimension with value > 1
        */
        FORCEINLINE bool isCommonVector(int& posOfNonUnityDim) const;


        /**
        *  returns true if array is scalar
        */
        FORCEINLINE bool isScalar() const;

        /**
        * Returns data type of this array
        * @return
        */
        FORCEINLINE DataType dataType() const;

        /**
         * This method returns true if value is from Integer space
         * @return
         */
        bool isZ() const;

        /**
         * This method returns true if array is from Real space
         * @return
         */
        bool isR() const;

        /**
         * This method returns true if array is from Boolean space
         * @return
         */
        bool isB() const;

        /**
         * This method returns true if array contains Complex numbers
         * @return
         */
        bool isC() const;

        /**
         * This method returns true if array contains String
         * @return
         */
        bool isS() const;

        /**
        *  inline accessing operator for matrix, i - absolute index
        */
        //FORCEINLINE NDArray operator()(const Nd4jLong i) const;

        /**
        *  inline modifying operator for matrix, i - absolute index
        */
        //FORCEINLINE NDArray& operator()(const Nd4jLong i);

        /**
        *  inline accessing operator for 2D array, i - row, j - column
        */
        //FORCEINLINE NDArray operator()(const Nd4jLong i, const Nd4jLong j) const;

        /**
        *  inline modifying operator for 2D array, i - row, j - column
        */
        //FORCEINLINE NDArray& operator()(const Nd4jLong i, const Nd4jLong j);

        /**
        *  inline accessing operator for 3D array, i - height, j - width, k - depth
        */
        //FORCEINLINE NDArray operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const;

        /**
        *  inline modifying operator for 3D array, i - height, j - width, k - depth
        */
        //FORCEINLINE NDArray& operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k);

        /**
        *  inline modifying operator for 4D array, i - height, j - width, k - depth
        */
        //FORCEINLINE NDArray& operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w);

        /**
        *  inline accessing operator for 4D array, i - height, j - width, k - depth
        */
        //FORCEINLINE NDArray operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) const;

        /**
        *  inline modifying operator for ND array
        *  idx - array with corresponding indexes, for example {2,10,0,5,...,8}, number of indexes should be equal to array rank
        */
        //FORCEINLINE NDArray& operator()(const Nd4jLong* idx);

        /**
        *  inline accessing operator for ND array
        *  idx - array with corresponding indexes, for example {2,10,0,5,...,8}, number of indexes should be equal to array rank
        */
        //FORCEINLINE NDArray operator()(const Nd4jLong* idx) const;



        template <typename T>
        std::vector<T> asVectorT();


        FORCEINLINE bool isAttached();

        NDArray* detach();

        FORCEINLINE bool operator==(const NDArray &other) const;

        FORCEINLINE bool operator!=(const NDArray &other) const;
    };




//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////
    bool NDArray::isAttached() {
        return this->_context->getWorkspace() != nullptr;
    }

    template <typename T, typename R>
    FORCEINLINE R NDArray::templatedGet(void *buffer, Nd4jLong index) const {
            auto b = reinterpret_cast<T*>(buffer);
            auto v = static_cast<R>(b[index]);
            return v;
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::setShapeInfo(Nd4jLong *shapeInfo) {
        auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(shapeInfo);
        _shapeInfo = buffer.primaryAsT<Nd4jLong>();
        _shapeInfoD = buffer.specialAsT<Nd4jLong>();

        if (shapeInfo != nullptr) {
            _dataType = ArrayOptions::dataType(_shapeInfo);
            if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                _length = 0;
            else
                _length = shape::length(_shapeInfo);
        }
        else {
            _dataType = nd4j::DataType::INHERIT;
            _length = 0;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::setShapeInfo(Nd4jLong *shapeInfo, const nd4j::DataType dtype) {
        auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(shapeInfo);
        _shapeInfo = buffer.primaryAsT<Nd4jLong>();
        _shapeInfoD = buffer.specialAsT<Nd4jLong>();

        if (shapeInfo != nullptr) {
            _dataType = dtype;
            if(ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                _length = 0;
            else
                _length = shape::length(_shapeInfo);
        }
        else {
            _dataType = nd4j::DataType::INHERIT;
            _length = 0;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    char NDArray::ordering() const {
        return shape::order(_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isView() const {
        return _isView;
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong* NDArray::shapeOf() const {
        return shape::shapeOf(_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong* NDArray::stridesOf() const {
        return shape::stride(_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    int NDArray::rankOf() const {
        return shape::rank(_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::lengthOf() const {
        return _length;
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::rows() const {
        if (this->rankOf() == 1)
            return 1;

        if (this->rankOf() > 2)
            throw std::runtime_error("Array with rank > 2 can't have rows");

        return shapeOf()[0];
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::columns() const {
        if (this->rankOf() == 1)
            return this->lengthOf();

        if (this->rankOf() > 2)
            throw std::runtime_error("Array with rank > 2 can't have columns");

        return shapeOf()[1];
    }

    //////////////////////////////////////////////////////////////////////////

    size_t NDArray::sizeOfT() const {

        return DataTypeUtils::sizeOfElement(_dataType);
    }

    //////////////////////////////////////////////////////////////////////////
    Nd4jLong NDArray::ews() const {
        if (this->isEmpty() || this->rankOf() == 0)
            return 1;

        return shape::elementWiseStride(_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::nonNull() const {
        if (isEmpty())
            return true;

        if(!Environment::getInstance()->isCPU())
            return getDataBuffer()->special() != nullptr && getSpecialShapeInfo() != nullptr;

        return getDataBuffer()->primary() != nullptr && getShapeInfo() != nullptr;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isMatrix() const {
        if (isEmpty())
            return false;

        return shape::isMatrix(this->_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isVector() const {
        if (isEmpty())
            return false;
        if (rankOf() == 1)
            return true;
        return !isScalar() && shape::isVector(this->_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isColumnVector() const {
        if (isEmpty())
            return false;

        return !isScalar() && shape::isColumnVector(this->_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isRowVector() const {
        if (isEmpty())
            return false;

        // 1D edge case
        if (shape::rank(this->_shapeInfo) == 1)
            return true;

        return !isScalar() && shape::isRowVector(this->_shapeInfo);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isCommonVector(int& posOfNonUnityDim) const {

        return shape::isCommonVector(_shapeInfo, posOfNonUnityDim);
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::isScalar() const {
        return shape::isScalar(this->_shapeInfo);
    }

//////////////////////////////////////////////////////////////////////////
// accessing operator for matrix, i - absolute index
/*
NDArray NDArray::operator()(const Nd4jLong i) const {

    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): input index is out of array length !");

    auto ews   = shape::elementWiseStride(_shapeInfo);
    char order = ordering();

    if(ews == 1 && order == 'c') {
        auto cast = reinterpret_cast<int8_t *>(_buffer) + (i * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
        return result;
    } else if(ews > 1 && order == 'c') {
        auto cast = reinterpret_cast<int8_t *>(_buffer) + (i * ews * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
        return result;
    } else {
        Nd4jLong idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());

        auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
        return result;
    }
}
*/
//////////////////////////////////////////////////////////////////////////
// modifying operator for matrix, i - absolute index
/*
NDArray& NDArray::operator()(const Nd4jLong i) {
    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): input index is out of array length !");

    auto ews = shape::elementWiseStride(_shapeInfo);
    auto order = ordering();

    if(ews == 1 && order == 'c') {
        auto cast = reinterpret_cast<int8_t *>(_buffer) + (i * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));

        // FIXME: bad
        return result;
    } else if(ews > 1 && order == 'c') {
        auto cast = reinterpret_cast<int8_t *>(_buffer) + (i * ews * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
        return result;
    } else {
        Nd4jLong idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());

        auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
        NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
        return result;
    }
}*/

//////////////////////////////////////////////////////////////////////////
// accessing operator for 2D matrix, i - row, j - column
/*
NDArray NDArray::operator()(const Nd4jLong i, const Nd4jLong j) const {

    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");

    Nd4jLong coords[2] = {i, j};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    // TODO: do we really want a view here?
    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
    return result;
}
*/
//////////////////////////////////////////////////////////////////////////
// modifying operator for 2D matrix, i - row, j - column
/*
NDArray& NDArray::operator()(const Nd4jLong  i, const Nd4jLong j) {
    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");

    Nd4jLong coords[2] = {i, j};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));

    //FIXME: bad, will crash!
    return result;
}
*/

//////////////////////////////////////////////////////////////////////////
// accessing operator for 3D array, i - row, j - column
/*
NDArray NDArray::operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {

    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || j >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");

    Nd4jLong coords[3] = {i, j, k};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
    return result;
}
*/

//////////////////////////////////////////////////////////////////////////
// modifying operator for 3D array
/*
NDArray& NDArray::operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) {

    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");

    Nd4jLong coords[3] = {i, j, k};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));

    //FIXME: bad, will crash!
    return result;
}
*/
/*
NDArray NDArray::operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) const {

    if (rankOf() != 4 || t >= shapeOf()[0] || u >= shapeOf()[1] || v >= shapeOf()[2] || w >= shapeOf()[3])
       throw std::invalid_argument("NDArray::operator(t,u,v,w): one of input indexes is out of array length or rank!=4 !");

    Nd4jLong coords[4] = {t, u, v, w};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
    return result;
}
*/
/*
NDArray& NDArray::operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) {

    if (rankOf() != 4 || t >= shapeOf()[0] || u >= shapeOf()[1] || v >= shapeOf()[2] || w >= shapeOf()[3])
       throw std::invalid_argument("NDArray::operator(t,u,v,w): one of input indexes is out of array length or rank!=4 !");

    Nd4jLong coords[4] = {t, u, v, w};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    // FIXME
    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
    return result;
}
*/
//////////////////////////////////////////////////////////////////////////
/*
NDArray NDArray::operator()(const Nd4jLong* idx) const {

    for(int i = 0; i < rankOf(); ++i)
        if (idx[i] >= sizeAt(i))
            throw std::invalid_argument("NDArray::operator(const Nd4jLong* idx): input index is out of dimension length !");

    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));
    return result;
}
*/
//////////////////////////////////////////////////////////////////////////
/*
NDArray& NDArray::operator()(const Nd4jLong* idx) {

    for(int i = 0; i < rankOf(); ++i)
        if (idx[i] >= sizeAt(i))
            throw std::invalid_argument("NDArray::operator(const Nd4jLong* idx): input index is out of dimension length !");

    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());

    auto cast = reinterpret_cast<int8_t *>(_buffer) + (xOffset * this->sizeOfT());
    NDArray result(cast, nd4j::ShapeBuilders::createScalarShapeInfo(this->dataType(), this->getWorkspace()));

    // FIXME
    return result;
}
*/


    //////////////////////////////////////////////////////////////////////////
    Nd4jLong FORCEINLINE NDArray::memoryFootprint() {
        Nd4jLong size = this->lengthOf() * this->sizeOfT();
        size += shape::shapeInfoByteLength(this->rankOf());
        return size;
    }

    //////////////////////////////////////////////////////////////////////////
    // still the definition of inline function must be in header file
    bool NDArray::isSameShape(const std::vector<Nd4jLong>& shape) const{
        if (this->isScalar() && shape.size() == 1 && shape[0] == 0)
            return true;
        if (this->rankOf() != (int) shape.size())
            return false;
        for (int e = 0; e < this->rankOf(); e++) {
            if (this->shapeOf()[e] != shape.at(e) && shape.at(e) != -1)
                return false;
        }
        return true;
    }

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(const NDArray *other) const {
    if (this->isEmpty() != other->isEmpty())
        return false;

    return isSameShape(std::vector<Nd4jLong>(other->_shapeInfo+1, other->_shapeInfo+1+other->_shapeInfo[0]));
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(const NDArray &other) const {
    return isSameShape(&other);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(const std::initializer_list<Nd4jLong>& other) const {
    return isSameShape(std::vector<Nd4jLong>(other));
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::areSameShapeAndType(const NDArray& other) const {

    if(rankOf() != other.rankOf() || _dataType != other._dataType)
        return false;

    for(int i = 0; i < rankOf(); ++i)
        if(sizeAt(i) != other.sizeAt(i))
            return false;

    return true;
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same _shapeInfo
// still the definition of inline function must be in header file

bool NDArray::isSameShapeStrict(const NDArray *other) const {
    return shape::equalsStrict(_shapeInfo, other->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isEmpty() const {
    if (this->_shapeInfo == nullptr)
        return false;

    return ArrayOptions::arrayType(this->getShapeInfo()) == ArrayType::EMPTY;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::operator==(const NDArray &other) const {
    // if (this->dataType() != other.dataType())    // this comparison is already present in equalsTo
    //         return false;

    if (!this->isSameShape(&other))
        return false;

    return this->equalsTo(&other);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::operator!=(const NDArray &other) const {
    if (this->dataType() != other.dataType())
        return true;

    if (!this->isSameShape(&other))
        return true;

    return !this->equalsTo(&other);
}

//////////////////////////////////////////////////////////////////////////
DataType NDArray::dataType() const {
    return _dataType;
    // return ArrayOptions::dataType(_shapeInfo);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T& NDArray::t(const Nd4jLong i) {

    // if (i >= _length)
    //     throw std::invalid_argument("NDArray::t(i): input index is out of array length !");
    if (DataTypeUtils::fromT<T>() != _dataType)
        throw std::invalid_argument("NDArray::t(i): type of array is not equal to template type T!");

    if(!isActualOnHostSide())
        syncToHost();

    tickWriteHost();
    return *(reinterpret_cast<T*>(bufferWithOffset(getOffset(i))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T& NDArray::t(const Nd4jLong i, const Nd4jLong j) {

    if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
            throw std::invalid_argument("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
    if (DataTypeUtils::fromT<T>() != _dataType)
        throw std::invalid_argument("NDArray::t(i,j): type of array is not equal to template type T!");

    if(!isActualOnHostSide())
        syncToHost();

    Nd4jLong coords[2] = {i, j};
    auto offset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    tickWriteHost();
    return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const Nd4jLong i) const {

    // if (i >= _length)
    //     throw std::invalid_argument("NDArray::t(i): input index is out of array length !");
    if (DataTypeUtils::fromT<T>() != _dataType)
        throw std::invalid_argument("NDArray::t(i): type of array is not equal to template type T!");

    if(!isActualOnHostSide())
        syncToHost();

    tickReadHost();
    return *(reinterpret_cast<T*>(bufferWithOffset(getOffset(i))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const Nd4jLong i, const Nd4jLong j) const {

    if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
            throw std::invalid_argument("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
    if (DataTypeUtils::fromT<T>() != _dataType)
        throw std::invalid_argument("NDArray::t(i,j): type of array is not equal to template type T!");

    if(!isActualOnHostSide())
        syncToHost();

    Nd4jLong coords[2] = {i, j};
    auto offset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    tickReadHost();
    return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
}

#ifndef __JAVACPP_HACK__
////////////////////////////////////////////////////////////////////////
std::shared_ptr<DataBuffer> NDArray::getDataBuffer() const {
    return _buffer;
}

////////////////////////////////////////////////////////////////////////
std::shared_ptr<DataBuffer> NDArray::dataBuffer() {
    return _buffer;
}
#endif

////////////////////////////////////////////////////////////////////////
void* NDArray::getBuffer() const {

    return _buffer->primary() != nullptr ? static_cast<int8_t*>(_buffer->primary()) + (_offset * sizeOfT()) : nullptr;
}

//////////////////////////////////////////////////////////////////////////
void* NDArray::buffer() {
    return _buffer->primary() != nullptr ? static_cast<int8_t*>(_buffer->primary()) + (_offset * sizeOfT()) : nullptr;
}

////////////////////////////////////////////////////////////////////////
Nd4jLong* NDArray::getShapeInfo() const {
    return _shapeInfo;
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong* NDArray::shapeInfo() {
    return _shapeInfo;
}

////////////////////////////////////////////////////////////////////////
Nd4jLong* NDArray::specialShapeInfo() {
    if (_shapeInfoD == nullptr)
        return _shapeInfo;
    // FIXME: this should be fixed once CUDA backend added
    return _shapeInfoD;
}

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::getBufferOffset() const {
    return _offset;
}

////////////////////////////////////////////////////////////////////////
Nd4jLong NDArray::bufferOffset() {
    return _offset;
}

////////////////////////////////////////////////////////////////////////
Nd4jLong* NDArray::getSpecialShapeInfo() const{
    if (_shapeInfoD == nullptr)
        return _shapeInfo;
    // FIXME: this should be fixed once CUDA backend added
    return _shapeInfoD;
}


#if defined(__CUDACC__) //&& defined(BUILD_TESTS)
// for CUDA we need stil stuff inline
#include "cuda/NDArrayLambda.hpp"
#endif

}

#endif
