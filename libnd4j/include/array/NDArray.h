/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#pragma  once
#include <array/ArrayOptions.h>
#include <array/ConstantShapeBuffer.h>
#include <array/DataBuffer.h>
#include <array/DataType.h>
#include <array/DataTypeUtils.h>
#include <array/ExtraArguments.h>
#include <array/InteropDataBuffer.h>
#include <array/ResultSet.h>
#include <array/ShapeDescriptor.h>
#include <execution/AffinityManager.h>
#include <graph/Intervals.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/shape.h>
#include <indexing/IndicesList.h>
#include <indexing/NDIndex.h>
#include <memory/MemoryCounter.h>
#include <ops/BroadcastBoolOpsTuple.h>
#include <ops/BroadcastIntOpsTuple.h>
#include <ops/BroadcastOpsTuple.h>
#include <stdint.h>
#include <system/op_enums.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <legacy/NativeOpExecutioner.h>
#include <types/float16.h>
#include <types/bfloat16.h>
#include <iostream>
namespace sd {
#ifndef __JAVACPP_HACK__
static void printFormatted(std::ostream& os, NDArray & arr, LongType depth, LongType limit);
//used in google test for printing
SD_LIB_EXPORT std::ostream& operator<<(std::ostream &os,  NDArray& arr);
void PrintTo(NDArray &arr, std::ostream *os);
#endif
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator+(NDArray &arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator+(NDArray &&arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator+( T scalar,  NDArray &arr);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator+( T scalar, NDArray &&arr);

template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator-( NDArray &arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator-(NDArray &&arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator-( T scalar,  NDArray &arr);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator-( T scalar, NDArray &&arr);

template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator*( NDArray &arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator*(NDArray &&arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator*( T scalar,  NDArray &arr);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator*( T scalar, NDArray &&arr);

template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator/( NDArray &arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator/(NDArray &&arr,  T scalar);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator/( T scalar,  NDArray &arr);
template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
SD_LIB_EXPORT NDArray operator/( T scalar, NDArray &&arr);

template <typename T1, typename T2,
    typename = typename std::enable_if<std::is_same<NDArray, typename std::decay<T1>::type>::value &&
                                       std::is_same<NDArray, typename std::decay<T2>::type>::value>::type>
SD_LIB_EXPORT NDArray operator+(T1 &&arr1, T2 &&arr2);
template <typename T1, typename T2,
    typename = typename std::enable_if<std::is_same<NDArray, typename std::decay<T1>::type>::value &&
                                       std::is_same<NDArray, typename std::decay<T2>::type>::value>::type>
SD_LIB_EXPORT NDArray operator-(T1 &&arr1, T2 &&arr2);
template <typename T1, typename T2,
    typename = typename std::enable_if<std::is_same<NDArray, typename std::decay<T1>::type>::value &&
                                       std::is_same<NDArray, typename std::decay<T2>::type>::value>::type>
SD_LIB_EXPORT NDArray operator*(T1 &&arr1, T2 &&arr2);
template <typename T1, typename T2,
    typename = typename std::enable_if<std::is_same<NDArray, typename std::decay<T1>::type>::value &&
                                       std::is_same<NDArray, typename std::decay<T2>::type>::value>::type>
SD_LIB_EXPORT NDArray operator/(T1 &&arr1, T2 &&arr2);

SD_LIB_EXPORT NDArray mmul(NDArray &, NDArray &);




class SD_LIB_EXPORT NDArray {
 private:
  NDArray(const NDArray &other);
  /**
   * This method applies given value to the buffer, wrt templates
   * @tparam T
   * @tparam Y
   * @param buffer
   * @param indices
   * @param value
   */
  template <typename T, typename Y>
  void templatedSet(void *buffer,  LongType *indices,  void *value);

  template <typename T, typename Y>
  void templatedSet(void *buffer,  LongType xOffset,  void *value);

  template <typename T>
  void templatedSet(void *buffer,  LongType xOfsset, DataType dtype,  void *value);

  template <typename T>
  void templatedAssign(void *xBuffer,  LongType xOffset,  void *yBuffer,
                       LongType yOffset);

  template <typename X, typename Y>
  void templatedDoubleAssign(void *xBuffer,  LongType xOffset,  void *yBuffer,
                             LongType yOffset);

  template <typename T, typename R>
  SD_INLINE R templatedGet(void  *buffer, const LongType index);

  template <typename T>
  void *templatedPointerShift(const LongType offset);

  SD_INLINE void copyBufferStatus(NDArray &other);

 protected:

  /**
   *  pointer on DataBuffer buffers in cpu/device memory
   */
  DataBuffer *_buffer = nullptr;



  /**
   *  contains shape info:  matrix rank, numbers of elements per each dimension, dimensions strides,
   * element-wise-stride, c-like or fortran-like order
   */

  ConstantShapeBuffer *_shapeInfoBuffer = nullptr;

  const LongType *_shapeInfo = nullptr;
  const LongType *_shapeInfoD = nullptr;

  /**
   *  pointer on device launch context (with all data needed there).
   */
  LaunchContext *_context = LaunchContext::defaultContext();

  // indicates if array's buffer is within workspace
  bool _isAttached = false;

  /**
   * Field to store cached length
   */
  LongType _length = -1L;

  LongType _offset = 0L;


  /**
   * deviceID where this NDArray belongs to
   */
  int _deviceId = AffinityManager::currentDeviceId();

  template <typename T>
  std::string*  toStringValue(T value);

 public:
  NDArray() = default;
#ifndef __JAVACPP_HACK__
#if defined(SD_GCC_FUNCTRACE)
  StackTrace creationTrace;
#endif
#endif

  /**
   *  do not allocate memory, memory for array is passed from outside
   */
#ifndef __JAVACPP_HACK__
  NDArray(DataBuffer *  buffer,  ShapeDescriptor *descriptor,
          LaunchContext *context = LaunchContext::defaultContext(), const LongType offset = 0);

  NDArray(DataBuffer *  buffer, const sd::LongType *shapeInfo,
          sd::LaunchContext *context = LaunchContext::defaultContext(), const sd::LongType offset = 0);

  NDArray(DataBuffer *  buffer, char order, std::vector<LongType> &shape,
          LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create scalar array containing string utf8
   *
   */
  NDArray(const char *str, DataType dtype = UTF8, LaunchContext *context = LaunchContext::defaultContext())
      : NDArray(std::string(str), dtype, context) {}
  NDArray(const std::string &string, DataType dtype = UTF8, LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create scalar array containing string utf16
   *
   */
  NDArray(const char16_t *u16string, DataType dtype = UTF16, LaunchContext *context = LaunchContext::defaultContext())
      : NDArray(std::u16string(u16string), dtype, context) {}

  NDArray(const std::u16string &u16string, DataType dtype = UTF16,
          LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create scalar array containing string utf32
   *
   */
  NDArray(const char32_t *u32string, DataType dtype = UTF32, LaunchContext *context = LaunchContext::defaultContext())
      : NDArray(std::u32string(u32string), dtype, context) {}

  NDArray(const std::u32string &u32string, DataType dtype = UTF32,
          LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create array from vector of utf8 strings
   *
   */
  NDArray(std::vector<LongType> &shape, const std::vector<const char *> &strings, DataType dtype = UTF8,
          LaunchContext *context = LaunchContext::defaultContext());
  NDArray(std::vector<sd::LongType> &shape, const std::vector<std::string> &string, const sd::DataType dataType = UTF8,
          sd::LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create array from vector of utf16 strings
   *
   */
  NDArray(std::vector<LongType> &shape, const std::vector<const char16_t *> &strings, DataType dtype = UTF16,
          LaunchContext *context = LaunchContext::defaultContext());
  NDArray(std::vector<LongType> &shape, const std::vector<std::u16string> &string, DataType dtype = UTF16,
          LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructors create array from vector of utf32 strings
   *
   */
  NDArray(std::vector<LongType> &shape, const std::vector<const char32_t *> &strings, DataType dtype = UTF32,
          LaunchContext *context = LaunchContext::defaultContext());
  NDArray(std::vector<sd::LongType> &shape, const std::vector<std::u32string> &string, sd::DataType dtype = UTF32,
          sd::LaunchContext *context = LaunchContext::defaultContext());

#endif

  /**
   *  do not allocate memory, memory for array is passed from outside
   */
  NDArray(void *buffer, const sd::LongType *shapeInfo, sd::LaunchContext *context, const bool isBuffAlloc,
          sd::LongType offset);

  /**
   *  do not allocate memory, memory for array is passed from outside
   *  we suppose the content of both (device and host) buffers is identical
   */
  NDArray(void *buffer, void *bufferD, const sd::LongType *shapeInfo, sd::LaunchContext *context,
          const bool isBuffAlloc, const bool isBuffDAlloc, sd::LongType offset);

  /**
   *  copy constructor
   */
  NDArray(NDArray &other);

  /**
   *  move constructor
   */
  NDArray(NDArray &&other) noexcept;

  /**
   *  constructor, create array stored at given workspace
   */
  NDArray(LaunchContext *context);

  /**
   *  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to zeros,
   * if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently
   */
  NDArray(const LongType *shapeInfo, bool copyStrides = false, LaunchContext *context = LaunchContext::defaultContext(), bool nullify = true);

  /**
   *  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to be
   * zeros, if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently set
   * dtype as array type
   */
  NDArray(const LongType *shapeInfo, DataType dtype, bool copyStrides = false,
          LaunchContext *context = LaunchContext::defaultContext(), bool nullify = true);

  /**
   *  this constructor creates new array using shape information contained in vector argument
   */
  NDArray(const char order, std::vector<sd::LongType> &shape, sd::DataType dtype = DOUBLE,
          sd::LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This constructor creates new array with elements copied from data and using shape information stored in shape,
   * elements from data will be casted to dtype
   */
  NDArray(char order, std::vector<LongType> &shape,  std::vector<double> &data, DataType dtype = DOUBLE,
          LaunchContext *context = LaunchContext::defaultContext());

  /**
   *  this constructor creates new array using given buffer (without memory allocation) and shape information stored in
   * shape
   */
  NDArray(void *buffer, char order, std::vector<LongType> &shape, DataType dtype,
          LaunchContext *context = LaunchContext::defaultContext(), const bool isBuffAlloc = false);



  /**
   * This method returns new array with the same shape & data type
   * @return
   */
  NDArray &like();

  /**
   * This method returns new uninitialized array with the same shape & data type
   * @return
   */
  NDArray &ulike();

  /**
   *  this constructor creates new NDArray with shape matching "other" array,
   *  doesn't copy "other" elements into new array !!!
   */
  explicit NDArray(NDArray *other, bool copyStrides = false,
                   LaunchContext *context = LaunchContext ::defaultContext());

  /**
   *  this constructor creates scalar(and set its value = 0) or empty array depending on bool argument isScalar
   */
  NDArray(DataType dtype, LaunchContext *context = LaunchContext::defaultContext(), bool isScalar = true);

  /**
   * This method blocks until asynchronous operation finishes
   */
  void synchronize(const char *msg);

  /**
   * This method allows to set _isAttached flag
   * @param reallyAttached
   */
  void setAttached(bool reallyAttached);

  void tickWriteHost();
  void tickWriteDevice();
  void tickReadHost();
  void tickReadDevice();
  void tickBothActual();
  bool isActualOnHostSide();
  bool isActualOnDeviceSide();
  void makeBothBuffersActual();

  void syncToHost();
  void syncToDevice();
  void syncShape();


  /**
   * This method can be used on architectures that use special buffers
   * @param writeList
   * @param readList
   */
  static void registerSpecialUse(const std::vector<NDArray *> &writeList,
                                 const std::vector<NDArray *> &readList = {});
  static void prepareSpecialUse(const std::vector<NDArray *> &writeList,
                                const std::vector<NDArray *> &readList = {}, bool synchronizeWritables = false);

  static void registerPrimaryUse(const std::vector<NDArray *> &writeList,
                                 const std::vector<NDArray *> &readList = {});
  static void preparePrimaryUse(const std::vector<NDArray *> &writeList,
                                const std::vector<NDArray *> &readList = {}, bool synchronizeWritables = false);


  /**
   * This method returns buffer pointer offset by given number of elements, wrt own data type
   * @param offset
   * @return
   */
  void  *bufferWithOffset(LongType offset);

  void const *specialBufferWithOffset(LongType offset);
  /**
   *  copy assignment operator
   *  in particular, when dataType() != other.dataType() and both shapes are the same, there will be allocation of new
   * _buffer and dataType() acquires other.dataType()
   */
  NDArray &operator=(NDArray &other);

  /**
   *  move assignment operator
   */
  NDArray &operator=(NDArray &&other) noexcept;


  /**
   *  assignment operator, assigns the same scalar to all array elements
   */
  template <typename T>
  NDArray &operator=(const T scalar);

  /**
   *   operators for memory allocation and deletion
   */
  void *operator new(size_t i);
  void operator delete(void *p);

  void setContext(LaunchContext *context);

  /**
   *  create a new array by replicating current array by repeats times along given dimension
   *  axis - axis along which to repeat elements
   *  repeats - number of repetitions
   */
  NDArray repeat(const int axis, const std::vector<LongType> &repeats);

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
   *  fill target array by repeating current array
   *  axis - axis along which to repeat elements
   *  repeats - vector containing numbers of repetition for elements at given axis
   */
  void repeat(const int axis, const std::vector<LongType> &repeats, NDArray &target);

  /**
   *  creates array which points on certain sub-range of this array, sub-range is defined by given indices
   */
  NDArray subarray(IndicesList &indices);
  NDArray subarray(const std::initializer_list<NDIndex *> &idx);
  NDArray subarray(const Intervals &idx);

  /**
   *  cast array elements to given dtype
   */
  NDArray cast(DataType dtype);

  void cast(NDArray &target, DataType dtype);

  /**
   *   returns _context
   */
  LaunchContext *getContext()  { return _context; }

#ifndef __JAVACPP_HACK__
  SD_INLINE DataBuffer * getDataBuffer();
  SD_INLINE DataBuffer *  dataBuffer();
#endif

  /**
   *   returns host buffer
   */
  SD_INLINE void *buffer();

  /**
   *   returns buffer offset (offset is the same for host and device buffers)
   */
  SD_INLINE LongType offset();

  /**
   *  checks if array has padded buffer
   */
  SD_INLINE bool hasPaddedBuffer();

  /**
   *  if _bufferD==nullptr return _buffer, else return _bufferD
   */
  void *specialBuffer();

  /**
   *   returns device buffer if compilation is for cuda case, otherwise returns host buffer
   */
  void *platformBuffer();

  template <typename T>
  T *bufferAsT();



  template <typename T>
  T *  bufferasTWithOffset(LongType offset);



  /**
   *   returns _shapeInfo
   */
  SD_INLINE const LongType *shapeInfo();


  /**
   *   returns _shapeInfo
   */
  SD_INLINE ConstantShapeBuffer *shapeInfoConstBuffer();


  SD_INLINE DataBuffer shapeInfoDataBuffer();
  /**
   * Returns True if it's legally empty NDArray, or false otherwise
   * @return
   */
  SD_INLINE bool isEmpty();

  /**
   *  if _shapeInfoD==nullptr return _shapeInfo, else return _shapeInfoD
   */
  SD_INLINE  const LongType *specialShapeInfo();


  /**
   *  permutes (in-place) the dimensions in array according to "dimensions" array
   */
  bool permutei(const std::initializer_list<LongType> &dimensions, const bool copyToNewBuff, const bool resetStrides);
  bool permutei(std::vector<LongType> &dimensions, const bool copyToNewBuff, const bool resetStrides);
  bool permutei(sd::LongType *dimensions, const int rank);


  bool isFinite();
  bool hasNaNs();
  bool hasInfs();

  void copyBuffersContinuouslyFrom(NDArray &other, size_t sizeToCopyInBytes = 0, LongType offsetThis = 0,
                                   LongType offsetOther = 0);

  /**
   *  permutes the dimensions in array according to "dimensions" array, new array points on _buffer of this array
   */
  NDArray &permute(std::vector<LongType> &dimensions, bool copyToNewBuff, bool resetStrides) &;

  NDArray &permute(LongType *dimensions, const int rank, const bool copyToNewBuff, const bool resetStrides) &;
  NDArray &permute(std::vector<LongType> &dimensions, const bool copyToNewBuff, const bool resetStrides) &&;
  NDArray &permute(LongType *dimensions, const int rank, const bool copyToNewBuff, const bool resetStrides) &&;

  void permute(LongType *dimensions, const int rank, NDArray &target, const bool resetStrides);

  /**
* This method streamlines given view or permuted array, and reallocates buffer
*/
  void streamline(char order = 'a');

  /**
   *  prints information about array shape
   *  msg - message to print out
   */
  void printShapeInfo(const char *msg = nullptr);

  /**
   *  prints buffer elements raw without using
   *  shape information but instead just the databuffer itself.
   *  msg - message to print out
   *  limit - number of array elements to print out
   *  sync - if true check whether host buffer is actual, if it is not then make it so
   */
  void printBufferRaw(const char *msg = nullptr, sd::LongType limit = -1, const bool sync = true);

  /**
   *  prints _buffer (if host = true) or _bufferD (if host = false) as it is, that is in current state without checking
   * buffer status
   */
  template <typename T>
  void printCurrentBuffer(const bool host = true, const char *msg = nullptr, const int precision = 1);

  /**
   *  prints buffer elements, takes into account offset between elements (element-wise-stride)
   *  msg - message to print out
   *  limit - number of array elements to print out
   */
  void printIndexedBuffer(const char *msg = nullptr, LongType limit = -1);

  std::string * asIndexedString(LongType limit = -1);
  std::string * asString(LongType limit = -1);

  /**
   *  this method assigns values of given array to this one
   */
  void assign(NDArray other, bool allowParallelism = true);



  /**
   *  this method assigns given value to all elements in array
   */
  template <typename T, typename = typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value>::type>
  void assign( T &value, bool allowParallelism = true);

  /**
   *  returns new copy of this array, optionally in different order
   */
  NDArray dup(const char newOrder = 'a', bool forceOriginalBuffer = false);



  /**
   *  returns sum of all elements of array
   */
  NDArray sumNumber();


  /**
   *  returns prod of all elements of array
   */
  NDArray prodNumber();


  /**
   *  returns mean number of array
   */
  NDArray meanNumber();

#ifndef __JAVACPP_HACK__

  /**
   * This method explicitly enforces new shape for this NDArray, old shape/stride information is lost
   */
  void enforce(const std::initializer_list<LongType> &dimensions, char order = 'a');
  void enforce(std::vector<LongType> &dimensions, char order = 'a');

  /**
   *  method reduces array by excluding its shapes along dimensions present in given dimensions vector, result is stored
   * in new array to be returned dimensions - array of dimensions to reduce along keepDims - if true then put unities in
   * place of reduced dimensions
   */

  NDArray reduceAlongDimension(reduce::FloatOps op, const std::vector<LongType> *dimensions,
                               const bool keepDims = false);
  NDArray reduceAlongDimension(reduce::FloatOps op, const std::initializer_list<LongType> *dimensions,
                               const bool keepDims = false);

  NDArray reduceAlongDimension(reduce::SameOps op, const std::vector<LongType> *dimensions,
                               const bool keepDims = false);
  NDArray reduceAlongDimension(reduce::SameOps op, const std::initializer_list<LongType> *dimensions,
                               const bool keepDims = false);

  NDArray reduceAlongDimension(reduce::BoolOps op, const std::vector<LongType> *dimensions,
                               const bool keepDims = false);
  NDArray reduceAlongDimension(reduce::BoolOps op, const std::initializer_list<LongType> *dimensions,
                               const bool keepDims = false);

  NDArray reduceAlongDimension(reduce::LongOps op, const std::vector<LongType> *dimensions,
                               const bool keepDims = false);
  NDArray reduceAlongDimension(reduce::LongOps op, const std::initializer_list<LongType> *dimensions,
                               const bool keepDims = false);

  /**
   *  method reduces array by excluding its shapes along dimensions present in given dimensions vector
   *  target - where to save result of reducing
   *  dimensions - array of dimensions to reduce along
   *  keepDims - if true then put unities in place of reduced dimensions
   *  extras - extra parameters
   */
  void reduceAlongDimension(reduce::FloatOps op, NDArray &target, const std::vector<LongType> *dimensions,
                            const bool keepDims = false, const bool checkTargetShape = true);
  void reduceAlongDimension(reduce::SameOps op, NDArray &target, const std::vector<LongType> *dimensions,
                            const bool keepDims = false, const bool checkTargetShape = true);
  void reduceAlongDimension(reduce::BoolOps op, NDArray &target, const std::vector<LongType> *dimensions,
                            const bool keepDims = false, const bool checkTargetShape = true);
  void reduceAlongDimension(reduce::LongOps op, NDArray &target, const std::vector<LongType> *dimensions,
                            const bool keepDims = false, const bool checkTargetShape = true);

  /**
   *  return variance of array elements set
   *  biasCorrected -  if true bias correction will be applied
   */
  NDArray varianceNumber(variance::Ops op, bool biasCorrected = true);

  /**
   *  apply scalar operation to array
   *  extraParams - extra parameters for operation
   *  returns scalar array
   */
  NDArray reduceNumber(reduce::FloatOps ops, void *extraParams = nullptr);
  NDArray reduceNumber(reduce::SameOps ops, void *extraParams = nullptr);
  NDArray reduceNumber(reduce::BoolOps ops, void *extraParams = nullptr);
  NDArray reduceNumber(reduce::LongOps ops, void *extraParams = nullptr);

  void reduceNumber(reduce::FloatOps ops, NDArray &target, void *extraParams = nullptr);
  void reduceNumber(reduce::SameOps ops, NDArray &target, void *extraParams = nullptr);
  void reduceNumber(reduce::BoolOps ops, NDArray &target, void *extraParams = nullptr);
  void reduceNumber(reduce::LongOps ops, NDArray &target, void *extraParams = nullptr);

  /**
   *  returns element index which corresponds to some condition imposed by operation
   *  extraParams - extra parameters for operation
   */
  NDArray indexReduceNumber(indexreduce::Ops op, ExtraArguments *extraParams = nullptr);

  /**
   *  returns index of max element in a given array (optionally: along given dimension(s))
   *  dimensions - optional vector with dimensions
   */
  LongType argMax(std::initializer_list<LongType> dimensions = {});

  // FIXME: remove this method eventually
  void makeBothActual()  {
    syncToDevice();
    syncToHost();
  }

  void applyTransform(transform::FloatOps op, NDArray &target, ExtraArguments *extraParams = nullptr);
  void applyTransform(transform::SameOps op, NDArray &target, ExtraArguments *extraParams = nullptr);
  void applyTransform(transform::AnyOps op, NDArray &target, ExtraArguments *extraParams = nullptr);
  void applyTransform(transform::BoolOps op, NDArray &target, ExtraArguments *extraParams = nullptr);
  void applyTransform(transform::StrictOps op, NDArray &target, ExtraArguments *extraParams = nullptr);

  /**
   *  apply OpName transformation to this array and store result in new array to be returned
   *  extraParams - extra parameters for operation
   */
  NDArray transform(transform::FloatOps op, void *extraParams = nullptr)  &;
  NDArray transform(transform::SameOps op, void *extraParams = nullptr)  &;
  NDArray transform(transform::BoolOps op, void *extraParams = nullptr)  &;
  NDArray transform(transform::StrictOps op, void *extraParams = nullptr)  &;
  NDArray transform(transform::FloatOps op, void *extraParams = nullptr) &&;
  NDArray transform(transform::SameOps op, void *extraParams = nullptr) &&;
  NDArray transform(transform::BoolOps op, void *extraParams = nullptr) &&;
  NDArray transform(transform::StrictOps op, void *extraParams = nullptr) &&;

  /**
   *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in this array
   *  other - second array necessary for pairwise operation
   *  extraParams - extra parameters for operation
   */
  void applyPairwiseTransform(pairwise::Ops op, NDArray &other, ExtraArguments *extraParams = nullptr);

  /**
   *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in target array
   *  other - second array necessary for pairwise operation
   *  target - where to store result
   *  extraParams - extra parameters for operation
   */
  void applyPairwiseTransform(pairwise::Ops op, NDArray &other, NDArray &target,
                              ExtraArguments *extraParams = nullptr);

  void applyPairwiseTransform(pairwise::BoolOps op, NDArray &other, NDArray &target,
                              ExtraArguments *extraParams = nullptr);

  void applyPairwiseTransform(pairwise::IntOps op, NDArray &other, NDArray &target,
                              ExtraArguments *extraParams = nullptr);


  bool isBroadcastableTo(NDArray &other);

  NDArray broadcastTo(const std::vector<LongType> & targetShape);

  /**
   *  apply operation which requires broadcasting, broadcast a smaller array (tad) along  bigger one (this)
   *  tad - array to broadcast
   *  dimensions -  dimensions array to broadcast along
   *  target - where to store result
   *  extraParams - extra parameters for operation
   */
  void applyBroadcast(broadcast::Ops op, const std::initializer_list<LongType> *dimensions, NDArray &tad,
                      NDArray &target, ExtraArguments *extraArgs = nullptr);

  void applyBroadcast(broadcast::Ops op, const std::vector<LongType> *dimensions, NDArray &tad, NDArray &target,
                      ExtraArguments *extraArgs = nullptr);

  void applyBroadcast(broadcast::BoolOps op, const std::vector<LongType> *dimensions, NDArray &tad,
                      NDArray &target, ExtraArguments *extraArgs = nullptr);

  void applyBroadcast(broadcast::IntOps op, const std::vector<LongType> *dimensions, NDArray &tad, NDArray &target,
                      ExtraArguments *extraArgs = nullptr);

  /**
   *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the
   * possibility of broadcasting other - input array extraParams - extra parameters for operation
   */
  NDArray applyTrueBroadcast(BroadcastOpsTuple op, NDArray &other,
                             ExtraArguments *extraArgs = nullptr) &;
  NDArray applyTrueBroadcast(BroadcastOpsTuple op, NDArray &&other, ExtraArguments *extraArgs = nullptr) &;
  NDArray applyTrueBroadcast(BroadcastOpsTuple op, NDArray &&other, ExtraArguments *extraArgs = nullptr) &&;
  NDArray applyTrueBroadcast(BroadcastOpsTuple op, NDArray &other, ExtraArguments *extraArgs = nullptr) &&;

  /**
   *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the
   * possibility of broadcasting other - input array target - where to store result checkTargetShape - if true check
   * whether target shape is suitable for broadcasting extraParams - extra parameters for operation
   */
  void applyTrueBroadcast(BroadcastOpsTuple op, NDArray &other, NDArray &target,
                          const bool checkTargetShape = true, ExtraArguments *extraArgs = nullptr);

  void applyTrueBroadcast(BroadcastBoolOpsTuple op, NDArray &other, NDArray &target,
                          const bool checkTargetShape = true, ExtraArguments *extraArgs = nullptr);

  void applyTrueBroadcast(BroadcastIntOpsTuple op, NDArray &other, NDArray &target,
                          const bool checkTargetShape = true, ExtraArguments *extraArgs = nullptr);

  /**
   *  apply a scalar operation to an array
   *  scalar - input scalar
   *  target - where to store result
   *  extraParams - extra parameters for operation
   */
  template <typename T>
  void applyScalar(scalar::Ops op, const T scalar, NDArray &target, ExtraArguments *extraParams = nullptr);

  template <typename T>
  void applyScalar(scalar::BoolOps op, const T scalar, NDArray &target,
                   ExtraArguments *extraParams = nullptr);

  template <typename T>
  void applyScalar(scalar::IntOps op, const T scalar, NDArray &target, ExtraArguments *extraParams = nullptr);

  /**
   *  apply a scalar operation to an array
   *  scalar - input array which is simple scalar
   *  target - where to store result
   *  extraParams - extra parameters for operation
   */
  void applyScalarArr(scalar::Ops op,  NDArray &scalar, NDArray &target,
                      ExtraArguments *extraParams = nullptr);

  void applyScalarArr(scalar::BoolOps op,  NDArray &scalar, NDArray &target,
                      ExtraArguments *extraParams = nullptr);

  void applyScalarArr(scalar::IntOps op,  NDArray &scalar, NDArray &target,
                      ExtraArguments *extraParams = nullptr);

#if defined(__CUDABLAS__)
  template <typename Lambda>
  SD_INLINE void applyLambda(Lambda func, NDArray &target);

  template <typename Lambda>
  SD_INLINE void applyPairwiseLambda(NDArray &other, Lambda func, NDArray &target);

  template <typename Lambda>
  SD_INLINE void applyIndexedLambda(Lambda func, NDArray &target);

  template <typename Lambda>
  SD_INLINE void applyIndexedPairwiseLambda(NDArray &other, Lambda func, NDArray &target);

  template <typename Lambda>
  SD_INLINE void applyTriplewiseLambda(NDArray &second, NDArray &third, Lambda func, NDArray &target);
#else

  /**
   *  apply operation "func" to an array
   *  func - what operation to apply
   *  target - where to store result
   */
  template <typename T>
  void applyLambda(const std::function<T(T)> &func, NDArray &target);

  /**
   *  apply pairwise operation "func" to an array
   *  other - input array
   *  func - what pairwise operation to apply
   *  target - where to store result
   */
  template <typename T>
  void applyPairwiseLambda(NDArray &other, const std::function<T(T, T)> &func, NDArray &target);

  template <typename T>
  void applyIndexedLambda(const std::function<T(sd::LongType, T)> &func, NDArray &target);

  template <typename T>
  void applyIndexedPairwiseLambda(NDArray &other, const std::function<T(sd::LongType, T, T)> &func, NDArray &target);

  template <typename T>
  void applyTriplewiseLambda(NDArray &second, NDArray &third, const std::function<T(T, T, T)> &func, NDArray &target);
#endif

  /**
   *  reduces dimensions in this array relying on index operation OpName
   *  dimensions - vector of dimensions to reduce along
   *  extraArgs - extra parameters for operation
   */
  NDArray applyIndexReduce(indexreduce::Ops op, const std::vector<LongType> *dimensions,
                           const ExtraArguments *extraParams = nullptr);

  /**
   *  reduces dimensions in array relying on index operation OpName
   *  target - where to store result
   *  dimensions - vector of dimensions to reduce along
   *  extraArgs - extra parameters for operation
   */
  void applyIndexReduce(indexreduce::Ops op, NDArray &target, const std::vector<LongType> *dimensions,
                        const ExtraArguments *extraParams = nullptr);

  /**
   *  apply reduce3 operation OpName to this and other array, return result in new output array
   *  other - input array
   *  extraArgs - extra parameters for operation
   */
  NDArray applyReduce3(reduce3::Ops op,  NDArray &other, const ExtraArguments *extraParams = nullptr);

  /**
   *  apply reduce3 operation OpName to this and other array, return result in new output array
   *  other - input array
   *  dimensions - vector of dimensions to reduce along (tads not axis)
   *  extraArgs - extra parameters for operation
   */
  NDArray applyAllReduce3(reduce3::Ops op,  NDArray &other, const std::vector<LongType> *dimensions,
                          const ExtraArguments *extraParams = nullptr);

  /**
   *  apply reduce3 (exec) operation OpName to this and other array, return result in new output array
   *  other - input array
   *  dimensions - vector of dimensions to reduce along (same as reduceAlongDimension)
   *  extraArgs - extra parameters for operation
   */
  NDArray applyReduce3(reduce3::Ops op,  NDArray &other, const std::vector<LongType> &dimensions,
                       const ExtraArguments *extraParams = nullptr);

  /**
   *  returns variance along given dimensions
   *  biasCorrected -  if true bias correction will be applied
   *  dimensions - vector of dimensions to calculate variance along
   */
  NDArray varianceAlongDimension(variance::Ops op, const bool biasCorrected,
                                 const std::vector<LongType> *dimensions);
  NDArray varianceAlongDimension(variance::Ops op, const bool biasCorrected,
                                 const std::initializer_list<LongType> *dimensions);

  void varianceAlongDimension(variance::Ops op, NDArray &target, const bool biasCorrected,
                              const std::vector<LongType> *dimensions);
  void varianceAlongDimension(variance::Ops op, NDArray &target, const bool biasCorrected,
                              const std::initializer_list<LongType> *dimensions);

#endif

  /**
   *   apply transpose operation to the copy of this array, that is this array remains unaffected
   */
  NDArray transpose() &;
  NDArray transpose() &&;

  /**
   *  perform transpose operation and store result in target, this array remains unaffected
   *  target - where to store result
   */
  void transpose(NDArray &target);

  /**
   *  apply in-place transpose operation to this array, so this array becomes transposed
   */
  void transposei();

  /**
   *  returns the number of arrays pointing on specified dimension(s)
   *  dimensions - array of dimensions to point on
   */
  LongType tensorsAlongDimension(std::initializer_list<LongType> dimensions);
  LongType tensorsAlongDimension(const std::vector<LongType> *dimensions);

  /**
   *  returns true if elements of two arrays are equal to within given epsilon value
   *  other - input array to compare
   *  eps - epsilon, this value defines the precision of elements comparison
   */
  bool equalsTo(NDArray *other, double eps = 1e-5);
  bool equalsTo(NDArray &other, double eps = 1e-5);

  /**
   *  add given row vector to all rows of this array
   *  row - row vector to add
   */
  void addiRowVector(NDArray &row);

  /**
   *  add given row vector to all rows of this array, store result in target
   *  row - row vector to add
   *  target - where to store result
   */
  void addRowVector(NDArray &row, NDArray &target);

  /**
   *  subtract given row vector from all rows of this array, store result in target
   *  row - row vector to subtract
   *  target - where to store result
   */
  void subRowVector(NDArray &row, NDArray &target);

  /**
   *  multiply all rows of this array on given row vector, store result in target
   *  row - row vector to multiply on
   *  target - where to store result
   */
  void mulRowVector(NDArray &row, NDArray &target);

  /**
   *  divide all rows of this array on given row vector, store result in target
   *  row - row vector to divide on
   *  target - where to store result
   */
  void divRowVector(NDArray &row, NDArray &target);

  /**
   *  add given column vector to all columns of this array, store result in target
   *  column - column vector to add
   *  target - where to store result
   */
  void addColumnVector(NDArray &column, NDArray &target);

  /**
   *  add given column vector to all columns of this array, this array becomes affected (in-place operation)
   *  column - column vector to add
   */
  void addiColumnVector(NDArray &column);

  /**
   *  multiply all columns of this array on given column vector, this array becomes affected (in-place operation)
   *  column - column vector to multiply on
   */
  void muliColumnVector(NDArray &column);

  /**
   *  returns number of bytes used by _buffer & _shapeInfo
   */
  SD_INLINE LongType memoryFootprint();

  /**
   *  these methods suited for FlatBuffers use
   */
  template <typename T>
  std::vector<T> getBufferAsVector();
  std::vector<LongType> getShapeAsVector();
  std::vector<sd::LongType> getStrideAsVector();
  std::vector<int> getShapeAsVectorInt();
  std::vector<LongType> getShapeInfoAsVector();
  std::vector<int64_t> getShapeInfoAsFlatVector();
  std::vector<int64_t> getShapeAsFlatVector();

  /**
   *  set new order and shape in case of suitable array length (in-place operation)
   *  order - order to set
   *  shape - shape to set
   *  copyToNewBuff - if true then old buffer will be copied to new buffer if last one will be allocated after reshaping
   *  if there was permute applied before or there are weird strides, then new buffer is allocated for array
   */
  bool reshapei(const char order, const std::initializer_list<sd::LongType> &shape);
  bool reshapei(const char order, const std::vector<sd::LongType> &shape);

  bool reshapei(const std::initializer_list<sd::LongType> &shape);
  bool reshapei(const std::vector<sd::LongType> &shape);

  void printStringInternalState();
  void printStringType();
  void checkIfStringArrayAndNotEmpty();

  void debugStringArray();
  /**
   *  creates new array with corresponding order and shape, new array will point on _buffer of this array
   *  order - order to set
   *  shape - shape to set
   *
   * if permute have been applied before or there are weird strides, then new buffer is allocated for new array
   */
  NDArray &reshape(char order, std::vector<sd::LongType> &shape, bool copyToNewBuff = true) &;
  NDArray &reshape(const char order, std::vector<sd::LongType> &shape, const bool copyToNewBuff = true) &&;

  /**
   *  calculate strides and set given order
   *  order - order to set
   */
  void updateStrides(const char order);

  NDArray *newShapeNoCopy(const std::vector<sd::LongType> &newShape, const char order);

  /**
   *  change an array by repeating it the number of times given by reps (in-place operation)
   *  repeats - contains numbers of repetitions
   */
  void tilei(const std::vector<LongType> &repeats);

  /**
   *  returns new array which is created by repeating of this array the number of times given by reps
   *  repeats - contains numbers of repetitions
   */
  NDArray tile(const std::vector<LongType> &repeats);

  /**
   *  change an array by repeating it the number of times given by reps (in-place operation)
   *  repeats - contains numbers of repetitions
   *  target - where to store result
   */
  void tile(const std::vector<LongType> &repeats, NDArray &target);

  /**
   *  change an array by repeating it the number of times to acquire the new shape which is the same as target shape
   *  target - where to store result
   */
  void tile(NDArray &target);

  /**
   *  check whether array is identity matrix
   */
  bool isIdentityMatrix();

  /**
   *  check whether array is unitary matrix
   */
  bool isUnitary();


  std::ostream& operator<<(std::ostream &os);

  /**
   *  operator returns subarray with buffer pointing at this->_buffer with offset defined by given intervals
   *  idx - intervals of indexes which define the subarrays to point on, idx has form {dim0Start,dim0End,
   * dim1Start,dim1End, ....} and length (2 * this->rankOf()) when (dimStart == dimEnd) then whole range will be used
   * for current dimension keepUnitiesInShape - if false then eliminate unities from resulting array shape, for example
   * {1,a,1,b} -> {a,b} isStrided - if true then idx has length (3 * this->rankOf()) and contains additional stride
   * numbers which correspond to stride between dimStart and dimEnd, so structure of idx is like
   * {dim0Start,dim0End,dim0Stride,    dim1Start,dim1End,dim1Stride, ....}
   */
  NDArray& operator()(const std::vector<LongType> &idx, const bool keepUnitiesInShape = false,
                      const bool isStrided = false);

  /**
   *  evaluates subarray with buffer pointing at this->_buffer and offset defined by given sequential index subArrIdx
   * and dimensions in dimsToExclude subArrIdx - index of current sub-array dimsToExclude - MUST BE SORTED, dimensions
   * to evaluate sub-array along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays
   * with shape [3,5], and subArrIdx must be in range [0,7] if dimsToExclude is empty then idxRanges containing all
   * zeros (means whole array) will be returned. keepUnitiesInShape - if false then eliminate unities from resulting
   * array shape, for example {1,a,1,b} -> {a,b}
   */
  NDArray& operator()(const LongType subArrIdx, const std::vector<LongType> &dimsToExclude,
                      bool keepUnitiesInShape = false);

  /**
   * processes whole set of sub-arrays
   * evaluates shapeInfo of sub-arrays (all sub-arrays have the same shapeInfo) and their buffer offsets (each sub-array
   * has its own unique offset from original this-buffer) dimsToExclude - MUST BE SORTED, dimensions to evaluate
   * sub-array along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays with shape
   * [3,5] if dimsToExclude.size() = array rank it means sub-array is whole array and copy of original_shapeInfo will be
   * returned and one zero offset subArrShapeInfo    - output argument, contains shapeInfo common for all sub-arrays
   * subArrOffsets      - output argument, contains successive sub-arrays offsets from original this-buffer
   * keepUnitiesInShape - if false then eliminate unities from sub-array shapeInfo, for example {1,a,1,b} -> {a,b}
   */
  void getSubArrShapeAndOffsets(const std::vector<LongType> &dimsToExclude, LongType *&subArrShapeInfo,
                                LongType *&subArrOffsets, bool keepUnitiesInShape = false);

  /**
   *  addition unary operator array += other
   *  other - input array to add
   */
  void operator+=(NDArray &other);

  /**
   *  subtraction unary operator array -= other
   *  other - input array to add
   */
  void operator-=(NDArray &other);

  template <typename T>
  void operator+=(const T other);

  template <typename T>
  void operator-=(const T other);

  /**
   *  negative operator, it changes sign of all array elements on opposite
   */
  NDArray operator-() &;
  NDArray operator-() &&;

  /**
   *  pairwise multiplication unary operator array *= other
   *  other - input array to multiply on
   */
  void operator*=(NDArray &other);

  /**
   *  multiplication unary operator array *= scalar
   *  scalar - input scalar to multiply on
   */
  template <typename T>
  void operator*=(const T scalar);

  /**
   *  pairwise division unary operator: array /= other
   *  other - input array to divide on
   */
  void operator/=(NDArray &other);

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
  friend NDArray mmul(NDArray &left, NDArray &right);

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
   *  PLEASE NOTE: method doesn't take into account the shapes of arrays, shapes may be different except one condition:
   * arrays lengths must be the same
   */
  void swapUnsafe(NDArray &other);

  /**
   *  return vector with buffer which points on corresponding diagonal elements of array
   *  type - means of vector to be returned: column ('c') or row ('r')
   */
  NDArray diagonal(const char type);

  /**
   * fill target matrix with given value in one or two directions from main diagonal:
   *   - down from main diagonal starting at subdiagonal number "lower" if direction = 'l' (down) or 'b' (both)
   *   - up from main diagonal starting at superdiagonal number "upper"if direction = 'u' (up) or 'b' (both)
   * direction - in what direction to fill matrix. There are 3 possible directions:
   *   'u' - fill up, mathematically this corresponds to lower triangular matrix, subdiagonal "lower" unaffected
   *   'l' - fill down, mathematically this corresponds to upper triangular matrix, superdiagonal "upper" remains
   * unaffected 'b' - fill in both directions, both "lower" and "upper" are taken into account rest of target elements
   * are equal to this array elements target and this array should have same shapes, except when this_rank = 1 (in that
   * case should be target_rank = 2)
   *
   * includeEdges handles the cases where we need to include edges (basically >= or <= 0 and edges of the triangle)
   */
  template <typename T>
  void fillAsTriangular(const float value, int lower, int upper, NDArray &target, const char direction = 'b',const bool includeEdges = true);

  /**
   *  change an array by repeating it the number of times in order to acquire new shape equal to the input shape
   *
   *  shape  - contains new shape to broadcast array to
   *  target - optional argument, if target != nullptr the resulting array will be placed in target, in opposite case
   * tile operation is done in place
   */
  NDArray tileToShape(const LongType *shapeInfo);
  void tileToShape(const std::vector<LongType> &shape, NDArray &target);
#ifndef __JAVACPP_HACK__
  void tileToShape(const std::initializer_list<LongType> &shape, NDArray &target);
#endif

  template <typename N>
  NDArray asT();

  template <typename S>
  NDArray asS();

  NDArray asT(DataType dtype);

  void linspace(const double start);

  void linspace(const double start, const double step);

  /**
   *  calculates the trace of an array, that is sum of elements on main diagonal = sum array[i, i, i, ...]
   */
  double getTrace();

  ResultSet multipleTensorsAlongDimension(const std::vector<LongType> &indices,
                                          const std::vector<LongType> &dimensions);

  ResultSet allTensorsAlongDimension(const std::initializer_list<LongType> &dimensions);

  ResultSet allTensorsAlongDimension(const std::vector<LongType> &dimensions);

  void printAllTensorsAlongDimension(const std::vector<LongType> &dimensions);
  void printAllTensorsAlongDimension(const std::initializer_list<LongType> &dimensions);
  void printTensorAlongDimension(LongType index,const std::vector<LongType> &dimensions);
  void printTensorAlongDimension(LongType index,const std::initializer_list<LongType> &dimensions);

  ResultSet allExamples();

  /**
   *  set _shapeInfo
   */
  void setShapeInfo(const LongType *shapeInfo);
  void setShapeInfo(ShapeDescriptor *descriptor);
  void setShapeInfo(const ConstantShapeBuffer *shapeBuffer);

  /**
   *  returns absolute offset which corresponds to given sequential index
   */
  LongType getOffset(const LongType i);

  /**
   *  returns reference on array element with given index
   */
  template <typename T>
  SD_INLINE T &r(LongType i);
  template <typename T>
  SD_INLINE T &r(const LongType i, const LongType j);
  template <typename T>
  SD_INLINE T &r(const LongType i, const LongType j, const LongType k);
  template <typename T>
  SD_INLINE T &r(const LongType i, const LongType j, const LongType k, const LongType w);

  /**
   *  returns array element with given index
   *  i - element index in array
   */
  template <typename T>
  SD_INLINE T t(const LongType i);
  template <typename T>
  SD_INLINE T t(const LongType i, const LongType j);
  template <typename T>
  SD_INLINE T t(const LongType i, const LongType j, const LongType k);
  template <typename T>
  SD_INLINE T t(const LongType i, const LongType j, const LongType k, const LongType w);


  ~NDArray();


  /**
   *  set _shapeInfo
   */

  /**
   *  returns the value of "dim" dimension
   */
  LongType sizeAt(const int dim);

  /**
   *  returns stride of "dim" dimension
   */
  LongType strideAt(const int dim);

  /**
   *  returns order of array
   */
  SD_INLINE char ordering();

  /**
   *  return _isView
   */
  SD_INLINE bool isView();

  /**
   *  returns shape portion of shapeInfo
   */
  SD_INLINE LongType *shapeOf();

  /**
   *  returns strides portion of shapeInfo
   */
  SD_INLINE LongType *stridesOf();

  /**
   *  returns rank of array
   */
  SD_INLINE int rankOf();

  /**
   *  returns length of array
   */
  SD_INLINE LongType lengthOf();

  /**
   *  returns number of rows in array
   */
  SD_INLINE LongType rows();

  /**
   *  returns number of columns in array
   */
  SD_INLINE LongType columns();

  /**
   *  returns size of array elements type
   */
  SD_INLINE size_t sizeOfT();

  /**
   *  returns element-wise-stride
   */
  SD_INLINE LongType ews();

  // returns true if arrays have same shape
  SD_INLINE bool isSameShape(NDArray *other);
  SD_INLINE bool isSameShape(NDArray &other);
  SD_INLINE bool isSameShape(const std::initializer_list<LongType> &shape);
  SD_INLINE bool isSameShape(const std::vector<LongType> &shape);
  SD_INLINE bool areSameShapeAndType(NDArray &other);

  /**
   *  returns true if these two NDArrays have same rank, dimensions, strides, ews and order
   */
  SD_INLINE bool isSameShapeStrict(NDArray &other);

  /**
   *  returns true if buffer && shapeInfo were defined (non nullptr)
   */
  SD_INLINE bool nonNull();


  /**
   *  returns array element with given index from linear buffer
   *  i - element index in array
   */
  template <typename T>
  T e(const LongType i);

  /**
   *  returns element with given indexes from 2D array
   *  i - number of row
   *  j - number of column
   */
  template <typename T>
  T e(const LongType i, const LongType j);

  /**
   *  returns element with given indexes from 3D array
   *  i - height
   *  j - width
   *  k - depth
   */
  template <typename T>
  T e(const LongType i, const LongType j, const LongType k);

  /**
   *  returns element with given indexes from DD array
   */
  template <typename T>
  T e(const LongType i, const LongType j, const LongType k, const LongType l);

  /**
   *  returns array-scalar containing element of this array with given index
   *  i - element index in array
   */
  NDArray e(const LongType i);

  /**
   *  assigns given scalar to array element by given index, regards array buffer as linear
   *  i - element index in array
   *  value - scalar value to assign
   */
  template <typename T>
  void p(const LongType i, const T value);

  void p(const LongType i, NDArray &value);

  /**
   *  assigns given scalar to 2D array element by given indexes
   *  i - number of row
   *  j - number of row
   *  value - scalar value to assign
   */
  template <typename T>
  void p(const LongType i, const LongType j, const T value);

  /**
   *  assigns given scalar to 3D array element by given indexes
   *  i - height
   *  j - width
   *  k - depth
   *  value - scalar value to assign
   */
  template <typename T>
  void p(const LongType i, const LongType j, const LongType k, const T value);

  template <typename T>
  void p(const LongType i, const LongType j, const LongType k, const LongType l, const T value);
  void p(const LongType i, const LongType j, const LongType k, const LongType l, NDArray &value);

  template <typename T>
  void pIdx(const LongType *indices, const T value);

  /**
   *  returns true if array is 2D
   */
  SD_INLINE bool isMatrix();

  /**
   *  returns true if array is vector
   */
  SD_INLINE bool isVector();

  /**
   *  returns true if array is column vector
   */
  SD_INLINE bool isColumnVector();

  /**
   *  returns true if array is row vector
   */
  SD_INLINE bool isRowVector();

  /**
   *  returns true if all dimensions of array except one are unities, for example: [1,1,n,1], [n,1,1], [n], ...
   *  posOfNonUnityDim - one dimension with value > 1
   */
  SD_INLINE bool isCommonVector(LongType &posOfNonUnityDim);

  /**
   *  returns true if array is scalar
   */
  SD_INLINE bool isScalar();

  /**
   * Returns data type of this array
   * @return
   */
  SD_INLINE DataType dataType();

  /**
   * This method returns true if value is from Integer space
   * @return
   */
  bool isZ();

  /**
   * This method returns true if array is from Real space
   * @return
   */
  bool isR();

  /**
   * This method returns true if array is from Boolean space
   * @return
   */
  bool isB();

  /**
   * This method returns true if array contains Complex numbers
   * @return
   */
  bool isC();

  /**
   * This method returns true if array contains String
   * @return
   */
  bool isS();

  template <typename T>
  std::vector<T> asVectorT();

  SD_INLINE bool isAttached();

  NDArray *detach();

  SD_INLINE bool operator==(NDArray &other);

  SD_INLINE bool operator!=(NDArray &other);
  NDArray(void *buffer, const char order, const std::vector<LongType> &shape, DataType dtype,
          LaunchContext *context, const bool isBuffAlloc, const bool isView, LongType offset);
#ifndef __JAVACPP_HACK__
  NDArray(DataBuffer *  buffer, const char order, const std::vector<LongType> &shape, DataType dtype,
          LaunchContext *context, const bool isBuffAlloc, const bool isView, LongType offset);
#endif

};

//////////////////////////////////////////////////////////////////////////
///// IMPLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////
bool NDArray::isAttached() { return this->_context->getWorkspace() != nullptr; }



//needed to avoid ambiguity with nvcc and pre defined bfloat16/float 16 conversion paths
//this method is used in lieu of constexrp to avoid a dependency on c++ 17
template <typename T, typename R>
struct TemplatedGetter {
  static R get(void  *buffer, LongType index) {
    if(buffer == nullptr)
      THROW_EXCEPTION("TemplatedGetter: Buffer is nullptr!");
    auto b = reinterpret_cast<T const *>(buffer);
    auto v = static_cast<R>(b[index]);
    return v;
  }
};

template <>
struct TemplatedGetter<bfloat16, float16> {
  static float16 get(void  *buffer, LongType index) {
    auto b = reinterpret_cast<bfloat16 const *>(buffer);
    float intermediate = static_cast<float>(b[index]);
    auto v = static_cast<float16>(intermediate);
    return v;
  }
};

template <typename T, typename R>
SD_INLINE R NDArray::templatedGet(void  *buffer, LongType index)  {
  return TemplatedGetter<T, R>::get(buffer, index);
}

//////////////////////////////////////////////////////////////////////////
char NDArray::ordering()  { return shape::order(_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
bool NDArray::isView()  { return shape::isViewConst(_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
LongType *NDArray::shapeOf()  { return shape::shapeOf(_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
LongType *NDArray::stridesOf()  { return shape::stride(_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
int NDArray::rankOf()  { return shape::rank(_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
LongType NDArray::lengthOf() {
  if(_length < 1) {
    this->_length = shape::length(this->_shapeInfo);
  }
  return _length;
}

//////////////////////////////////////////////////////////////////////////
LongType NDArray::rows()  {
  if (this->rankOf() == 1) return 1;

  if (this->rankOf() > 2) THROW_EXCEPTION("Array with rank > 2 can't have rows");

  return shapeOf()[0];
}

//////////////////////////////////////////////////////////////////////////
LongType NDArray::columns()  {

  if (this->rankOf() == 1) {
    auto thisRef = const_cast<NDArray *>(this);
    return thisRef->lengthOf();
  }

  if (this->rankOf() > 2) THROW_EXCEPTION("Array with rank > 2 can't have columns");

  return shapeOf()[1];
}

//////////////////////////////////////////////////////////////////////////

size_t NDArray::sizeOfT()  { return DataTypeUtils::sizeOfElement(dataType()); }

//////////////////////////////////////////////////////////////////////////
LongType NDArray::ews()  {
  if (this->isEmpty() || this->rankOf() == 0) return 1;

  return shape::elementWiseStride(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::nonNull()  {
  if (isEmpty()) return true;

  if (!Environment::getInstance().isCPU())
    return getDataBuffer()->special() != nullptr && specialShapeInfo() != nullptr;

  return getDataBuffer()->primary() != nullptr && shapeInfo() != nullptr;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isMatrix()  {
  if (isEmpty()) return false;

  return 0 != shape::isMatrix(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isVector()  {
  if (isEmpty()) return false;
  if (rankOf() == 1) return true;
  return !isScalar() && shape::isVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isColumnVector()  {
  if (isEmpty()) return false;

  return !isScalar() && shape::isColumnVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isRowVector()  {
  if (isEmpty()) return false;

  // 1D edge case
  if (shape::rank(this->_shapeInfo) == 1) return true;

  return !isScalar() && shape::isRowVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isCommonVector(LongType &posOfNonUnityDim)  {
  return shape::isCommonVector(_shapeInfo, posOfNonUnityDim);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isScalar()  { return 0 != shape::isScalar(this->_shapeInfo); }

//////////////////////////////////////////////////////////////////////////
LongType SD_INLINE NDArray::memoryFootprint() {
  int len = isScalar() ? 1 : lengthOf();
  LongType size = len * this->sizeOfT();
  size += shape::shapeInfoByteLength(this->rankOf());
  return size;
}

//////////////////////////////////////////////////////////////////////////
// still the definition of inline function must be in header file
bool NDArray::isSameShape(const std::vector<LongType> &shape)  {
  if (this->isScalar() && shape.size() == 1 && shape[0] == 0) return true;
  if (this->rankOf() != (int)shape.size()) return false;
  for (int e = 0; e < this->rankOf(); e++) {
    if (this->shapeOf()[e] != shape[e] && shape[e] != -1) return false;
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(NDArray *other)  {
  if (this->isEmpty() != other->isEmpty()) return false;

  return isSameShape(std::vector<LongType>(other->_shapeInfo + 1, other->_shapeInfo + 1 + other->_shapeInfo[0]));
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(NDArray &other)  { return isSameShape(&other); }

//////////////////////////////////////////////////////////////////////////
bool NDArray::isSameShape(const std::initializer_list<LongType> &other)  {
  return isSameShape(std::vector<LongType>(other));
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::areSameShapeAndType(NDArray &other)  {
  if (rankOf() != other.rankOf() || dataType() != other.dataType()) return false;

  for (int i = 0; i < rankOf(); ++i)
    if (sizeAt(i) != other.sizeAt(i)) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same _shapeInfo
// still the definition of inline function must be in header file

bool NDArray::isSameShapeStrict(NDArray &other)  {
  return shape::equalsStrict(_shapeInfo, other._shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isEmpty()  {
  if (this->_shapeInfo == nullptr) THROW_EXCEPTION("NDArray::isEmpty() - shapeInfo is nullptr!");
  if(this->_shapeInfo[0] > SD_MAX_RANK || this->_shapeInfo[0] < 0) {
    std::string errorMessage;
    errorMessage += "NDArray::isEmpty() - rank of array is out of range! Shape info could have been deallocated. ";
    errorMessage += "Rank: ";
    errorMessage += std::to_string(this->_shapeInfo[0]);
    errorMessage += " Max rank: ";
    errorMessage += std::to_string(SD_MAX_RANK);
    errorMessage += " Min rank: ";
    errorMessage += std::to_string(0);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  bool baseEmpty =  ArrayOptions::hasPropertyBitSet(this->_shapeInfo, ARRAY_EMPTY);
  return baseEmpty;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::operator==(NDArray &other)  {
  auto constThis = const_cast<NDArray *>(this);
  auto constOther = const_cast<NDArray *>(&other);
  if (!constThis->isSameShape(constOther)) {
    return false;
  }

  return this->equalsTo(&other);
}

//////////////////////////////////////////////////////////////////////////

bool NDArray::operator!=(NDArray &other)  {
  auto constThis = const_cast<NDArray *>(this);
  auto constOther = const_cast<NDArray *>(&other);
  if (this->dataType() != constOther->dataType()) return true;

  if (!constThis->isSameShape(constOther)) return true;

  return !this->equalsTo(&other);
}

//////////////////////////////////////////////////////////////////////////
DataType NDArray::dataType()  {
  if(_shapeInfo == nullptr) {
    THROW_EXCEPTION("NDArray::dataType: shapeInfo is nullptr!");
  }
  return ArrayOptions::dataType(_shapeInfo);
}


////////////////////////////////////////////////////////////////////////
template <typename T>
T &NDArray::r(LongType i) {
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    sd_printf("Expected data type was %d but was %d\n", dataType(), inputDtype);
    THROW_EXCEPTION("NDArray::t(i): type of array is not equal to template type T!");
  }
  syncToHost();
  tickWriteHost();

  return *(reinterpret_cast<T *>(bufferWithOffset(getOffset(i))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T &NDArray::r(const LongType i, const LongType j) {
  if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
    THROW_EXCEPTION("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    sd_printf("Expected data type was %d but was %d\n", dataType(), inputDtype);
    THROW_EXCEPTION("NDArray::t(i,j): type of array is not equal to template type T!");
  }
  syncToHost();
  tickWriteHost();


  return *(reinterpret_cast<T *>(bufferWithOffset(i * strideAt(0) + j * strideAt(1))));
}

template <typename T>
T &NDArray::r(const LongType i, const LongType j, const LongType k) {
  if (rankOf() != 3 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2))
    THROW_EXCEPTION("NDArray::t(i,j,k): one of input indexes is out of array length or rank!=3!");
  if (DataTypeUtils::fromT<T>() != dataType())
    THROW_EXCEPTION("NDArray::t(i,j,k): type of array is not equal to template type T!");

  syncToHost();
  tickWriteHost();

  return *(reinterpret_cast<T *>(bufferWithOffset(i * strideAt(0) + j * strideAt(1) + k * strideAt(2))));
}

template <typename T>
T &NDArray::r(const LongType i, const LongType j, const LongType k, const LongType w) {
  if (rankOf() != 4 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2) || w >= sizeAt(3))
    THROW_EXCEPTION("NDArray::t(i,j,k,w): one of input indexes is out of array length or rank!=4 !");
  if (DataTypeUtils::fromT<T>() != dataType())
    THROW_EXCEPTION("NDArray::t(i,j,k,w): type of array is not equal to template type T!");

  syncToHost();
  tickWriteHost();

  return *(
      reinterpret_cast<T *>(bufferWithOffset(i * strideAt(0) + j * strideAt(1) + k * strideAt(2) + w * strideAt(3))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const LongType i)  {
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    sd_printf("Expected data type was %d but was %d\n", dataType(), inputDtype);
    THROW_EXCEPTION("NDArray::t(i): type of array is not equal to template type T!");
  }

  syncToHost();


  return *(reinterpret_cast<const T *>(bufferWithOffset(getOffset(i))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const LongType i, const LongType j)  {
  if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
    THROW_EXCEPTION("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    sd_printf("Expected data type was %d but was %d\n", dataType(), inputDtype);
    THROW_EXCEPTION("NDArray::t(i,j): type of array is not equal to template type T!");
  }
  syncToHost();

  return *(reinterpret_cast<const T *>(bufferWithOffset(i * strideAt(0) + j * strideAt(1))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const LongType i, const LongType j, const LongType k)  {
  if (rankOf() != 3 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2))
    THROW_EXCEPTION("NDArray::t(i,j,k): one of input indexes is out of array length or rank!=3!");
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    sd_printf("Expected data type was %d but was %d\n", dataType(), inputDtype);
    THROW_EXCEPTION("NDArray::t(i,j,k): type of array is not equal to template type T!");
  }
  syncToHost();

  return *(reinterpret_cast<const T *>(bufferWithOffset(i * strideAt(0) + j * strideAt(1) + k * strideAt(2))));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::t(const LongType i, const LongType j, const LongType k, const LongType w)  {
  if (rankOf() != 4 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2) || w >= sizeAt(3))
    THROW_EXCEPTION("NDArray::t(i,j,k,w): one of input indexes is out of array length or rank!=4!");
  auto inputDtype = DataTypeUtils::fromT<T>();
  if (inputDtype != dataType()) {
    std::string errorMessage;
    errorMessage += "Expected data type was ";
    errorMessage += DataTypeUtils::asString(inputDtype);
    errorMessage += " but was ";
    errorMessage += DataTypeUtils::asString(dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  syncToHost();

  return *(reinterpret_cast<const T *>(
      bufferWithOffset(i * strideAt(0) + j * strideAt(1) + k * strideAt(2) + w * strideAt(3))));
}

#ifndef __JAVACPP_HACK__
////////////////////////////////////////////////////////////////////////
DataBuffer * NDArray::getDataBuffer()  { return _buffer; }

////////////////////////////////////////////////////////////////////////
DataBuffer * NDArray::dataBuffer() { return _buffer; }
#endif

////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
template <typename T>
void * _bufferWithOffset(LongType offset,DataBuffer *buffer) {
 return reinterpret_cast<void *>(buffer->primaryAtOffset<T>(offset));
}

//note this is meant to be used with primary() (host side/cpu) use specialBuffer() for device side buffers
void *NDArray::buffer() {
  BUILD_SINGLE_SELECTOR(dataType(), return _bufferWithOffset, (offset(),getDataBuffer()),SD_COMMON_TYPES);
}

//////////////////////////////////////////////////////////////////////////
const LongType *NDArray::shapeInfo()  { return _shapeInfo; }



ConstantShapeBuffer * NDArray::shapeInfoConstBuffer()   { return _shapeInfoBuffer; }

DataBuffer NDArray::shapeInfoDataBuffer()   {
  auto primary = _shapeInfoBuffer->primary();
  auto voidPointer = const_cast<LongType *>(primary);
  auto void2 = reinterpret_cast<void *>(voidPointer);
  DataBuffer ret(void2, INT64, shape::shapeInfoByteLength(_shapeInfo[0]));
  return ret;

}


////////////////////////////////////////////////////////////////////////
const LongType *NDArray::specialShapeInfo()  {
  if (_shapeInfoD == nullptr) return _shapeInfo;
  // FIXME: this should be fixed once CUDA backend added
  return _shapeInfoD;
}

////////////////////////////////////////////////////////////////////////
LongType NDArray::offset()  { return _offset; }



////////////////////////////////////////////////////////////////////////
bool NDArray::hasPaddedBuffer()  { return ArrayOptions::hasPaddedBuffer(_shapeInfo); }



}  // namespace sd

#endif
