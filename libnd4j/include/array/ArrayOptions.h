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

//
// @author raver119@gmail.com
//

#ifndef ND4J_ARRAY_OPTIONS_H
#define ND4J_ARRAY_OPTIONS_H

#pragma once

#include <system/common.h>
#include <system/op_boilerplate.h>
#include <array/ArrayType.h>
#include <array/DataType.h>
#include <array/SpaceType.h>
#include <array/SparseType.h>

#include <initializer_list>
#include <vector>


//notice how each flag value is multiplied by 2
//if they are too close in value values will clash.
#define ARRAY_SPARSE 2
#define ARRAY_COMPRESSED 4
#define ARRAY_EMPTY 8
#define ARRAY_RAGGED 16

#define ARRAY_CSR 32
#define ARRAY_CSC 64
#define ARRAY_COO 128

// complex values
#define ARRAY_COMPLEX 512

// quantized values
#define ARRAY_QUANTIZED 1024

//  16 bit float FP16
#define ARRAY_HALF 4096

//  16 bit bfloat16
#define ARRAY_BHALF 2048

// regular 32 bit float
#define ARRAY_FLOAT 8192

// regular 64 bit float
#define ARRAY_DOUBLE 16384

// 8 bit integer
#define ARRAY_CHAR 32768

// 16 bit integer
#define ARRAY_SHORT 65536

// 32 bit integer
#define ARRAY_INT 131072

// 64 bit integer
#define ARRAY_LONG 262144

// boolean values
#define ARRAY_BOOL 524288

// UTF values
#define ARRAY_UTF8 1048576
#define ARRAY_UTF16 4194304
#define ARRAY_UTF32 16777216

// flag for extras
#define ARRAY_EXTRAS 2097152

// flag for signed/unsigned integers
#define ARRAY_UNSIGNED 8388608

// flag for arrays with padded buffer
#define ARRAY_HAS_PADDED_BUFFER (1 << 25)
//flags for when array has a view or not
#define ARRAY_IS_VIEW 33554432

//flag for when array needs a copy
//this is mainly used in the reshape_no_copy op but could be used elsewhere
#define ARRAY_NEEDS_COPY 67108864


//we need this in order to preserve the offset of the original buffer when creating the output array
//when views are created, sometimes we need to use the original offset of the array
//we don't need this very often and we don't store the offset in the shape info
//this preserves the offsets only being in the ndarray but allowing us to pass information
//when creating views, each flag is for an input in to an op, most of the time we only need the first 3
//but may need more. For now only the first one is used but may be needed elsewhere.
#define ARRAY_COPY_OFFSET_INPUT_0 134217728
#define ARRAY_COPY_OFFSET_INPUT_1 268435456
#define ARRAY_COPY_OFFSET_INPUT_2 536870912
#define ARRAY_COPY_OFFSET_INPUT_3 1073741824
#define ARRAY_COPY_OFFSET_INPUT_4 2147483648
#define ARRAY_COPY_OFFSET_INPUT_5 4294967296
#define ARRAY_COPY_OFFSET_INPUT_6 8589934592
#define ARRAY_COPY_OFFSET_INPUT_7 17179869184
#define ARRAY_COPY_OFFSET_INPUT_8 34359738368
#define ARRAY_COPY_OFFSET_INPUT_9 68719476736
#define ARRAY_COPY_OFFSET_INPUT_10 137438953472


#define DEFAULT_FLAG 0




namespace sd {
class SD_LIB_EXPORT ArrayOptions {
 public:
  static SD_HOST SD_INLINE  LongType extra(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  void setExtra(LongType *shapeInfo, LongType value);
  static SD_HOST SD_INLINE  bool isNewFormat(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  bool hasPropertyBitSet(const LongType *shapeInfo, LongType property);
  static SD_HOST SD_INLINE  bool togglePropertyBit(LongType *shapeInfo, LongType property);
  static SD_HOST SD_INLINE  void unsetPropertyBit(LongType *shapeInfo, LongType property);
  static SD_HOST SD_INLINE  void validateSingleDataType(LongType property);
  static SD_HOST SD_INLINE  void setPropertyBit(LongType *shapeInfo, LongType property);
  static SD_HOST SD_INLINE  void setPropertyBits(LongType *shapeInfo, std::initializer_list<LongType> properties);
  static SD_HOST SD_INLINE  sd::LongType numDataTypesSet(sd::LongType property);
  static SD_HOST SD_INLINE  bool isUnsigned(LongType *shapeInfo);
  static SD_HOST SD_INLINE  bool isSparseArray(sd::LongType *shapeInfo);
  static SD_HOST SD_INLINE  DataType dataType(const LongType *shapeInfo);

  static SD_HOST SD_INLINE  SpaceType spaceType(LongType *shapeInfo);
  static  SD_INLINE SD_HOST_DEVICE SpaceType spaceType(const LongType *shapeInfo);

  static SD_HOST SD_INLINE  ArrayType arrayType(LongType *shapeInfo);
  static SD_HOST SD_INLINE  ArrayType arrayType(const LongType *shapeInfo);

  static SD_HOST SD_INLINE  bool isView(LongType *shapeInfo);
  static SD_HOST SD_INLINE  void toggleIsView(LongType *shapeInfo);

  static  SD_INLINE SD_HOST_DEVICE SparseType sparseType(LongType *shapeInfo);
  static SD_HOST SD_INLINE  SparseType sparseType(const LongType *shapeInfo);

  static  SD_INLINE SD_HOST_DEVICE bool hasExtraProperties(LongType *shapeInfo);

  static SD_HOST SD_INLINE  bool hasPaddedBuffer(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  void flagAsPaddedBuffer(LongType *shapeInfo);

  static SD_HOST SD_INLINE  void resetDataType(LongType *shapeInfo);
  static SD_HOST SD_INLINE  LongType propertyWithoutDataType(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  void setDataType(LongType *shapeInfo, const DataType dataType);
  static SD_HOST SD_INLINE  LongType setDataTypeValue(LongType extraStorage, const DataType dataType);
  static SD_HOST SD_INLINE  LongType flagForDataType(const DataType dataType);
  static SD_HOST SD_INLINE  void copyDataType(LongType *to, const LongType *from);
  static SD_HOST SD_INLINE  const char *enumerateSetFlags(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  void unsetAllFlags(LongType *shapeInfo);
  static SD_HOST SD_INLINE  int enumerateSetFlags(const LongType *shapeInfo, const char **setFlagsOutput, int maxFlags);
  static SD_HOST SD_INLINE  const char *findFlagString(int flag);
  static SD_HOST SD_INLINE  LongType extraIndex(const LongType *shapeInfo);
  static SD_HOST SD_INLINE  LongType extraIndex(LongType *shapeInfo);
  static SD_HOST SD_INLINE  void unsetAllFlags(LongType &flagStorage);
  static SD_HOST SD_INLINE  const char *enumerateSetFlagsForFlags(const LongType flagStorage);
  static SD_HOST SD_INLINE  SpaceType spaceTypeForFlags(const LongType &flagStorage);
  static SD_HOST SD_INLINE  ArrayType arrayTypeForFlags(const LongType &flagStorage);
  static SD_HOST SD_INLINE  bool togglePropertyBitForFlags(LongType &flagStorage, LongType property);
  static SD_HOST SD_INLINE  LongType unsetPropertyBitForFlags(LongType &flagStorage, LongType property);
  static SD_HOST SD_INLINE  SparseType sparseTypeForFlags(const LongType &flagStorage);
  static SD_INLINE LongType setPropertyBitForFlagsValue(LongType extraStorage, LongType property);
  static SD_HOST SD_INLINE  bool hasPropertyBitSet(const LongType extra, LongType property);
  static SD_HOST SD_INLINE  void resetFlags(LongType *to);
  static SD_HOST SD_INLINE  LongType defaultFlag();

  static SD_HOST SD_INLINE   LongType propertyWithoutDataTypeValue(LongType extra);
  static SD_HOST SD_INLINE  DataType dataTypeValue(LongType property);
  static SD_INLINE bool isEmpty(LongType *shapeInfo);
  static SD_INLINE void toggleIsEmpty(LongType *shapeInfo);

  static SD_INLINE bool arrayNeedsCopy(LongType *shapeInfo);
  static SD_INLINE void toggleArrayNeedsCopy(LongType *shapeInfo);
};

}
#endif  // ND4J_ARRAY_OPTIONS_H :)
