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





namespace sd {
class SD_LIB_EXPORT ArrayOptions {
 public:
  static SD_HOST_DEVICE LongType extra(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE void setExtra(sd::LongType *shapeInfo, sd::LongType value);
  static SD_HOST_DEVICE bool isNewFormat(const sd::LongType *shapeInfo);
  static SD_HOST_DEVICE bool hasPropertyBitSet(const sd::LongType *shapeInfo, int property);
  static SD_HOST_DEVICE bool togglePropertyBit(sd::LongType *shapeInfo, int property);
  static SD_HOST_DEVICE void unsetPropertyBit(sd::LongType *shapeInfo, int property);

  static SD_HOST_DEVICE void setPropertyBit(sd::LongType *shapeInfo, int property);
  static SD_HOST_DEVICE void setPropertyBits(sd::LongType *shapeInfo, std::initializer_list<int> properties);

  static SD_HOST_DEVICE bool isSparseArray(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE bool isUnsigned(sd::LongType *shapeInfo);

  static sd::DataType dataType(const sd::LongType *shapeInfo);

  static SD_HOST_DEVICE SpaceType spaceType(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE SpaceType spaceType(const sd::LongType *shapeInfo);

  static SD_HOST_DEVICE ArrayType arrayType(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE ArrayType arrayType(const sd::LongType *shapeInfo);

  static SD_HOST_DEVICE SparseType sparseType(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE SparseType sparseType(const sd::LongType *shapeInfo);

  static SD_HOST_DEVICE bool hasExtraProperties(sd::LongType *shapeInfo);

  static SD_HOST_DEVICE bool hasPaddedBuffer(const sd::LongType *shapeInfo);
  static SD_HOST_DEVICE void flagAsPaddedBuffer(sd::LongType *shapeInfo);

  static SD_HOST_DEVICE void resetDataType(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE sd::LongType propertyWithoutDataType(const sd::LongType *shapeInfo);
  static SD_HOST_DEVICE void setDataType(sd::LongType *shapeInfo, const sd::DataType dataType);

  static SD_HOST_DEVICE void copyDataType(sd::LongType *to, const sd::LongType *from);
  static SD_HOST_DEVICE std::vector<std::string> enumerateSetFlags(const LongType *shapeInfo);
  static SD_HOST_DEVICE void unsetAllFlags(LongType *shapeInfo);
  static SD_HOST_DEVICE int enumerateSetFlags(const LongType *shapeInfo, const char **setFlagsOutput, int maxFlags);
  static SD_HOST_DEVICE const char *findFlagString(int flag);
  static SD_HOST_DEVICE sd::LongType extraIndex(const sd::LongType *shapeInfo);
  static SD_HOST_DEVICE sd::LongType extraIndex(sd::LongType *shapeInfo);
  static SD_HOST_DEVICE void unsetAllFlags(LongType &flagStorage);
  static SD_HOST_DEVICE int enumerateSetFlagsForFlags(const LongType &flagStorage, const char **setFlagsOutput,
                                                      int maxFlags);
  static SD_HOST_DEVICE SpaceType spaceTypeForFlags(const LongType &flagStorage);
  static SD_HOST_DEVICE ArrayType arrayTypeForFlags(const LongType &flagStorage);
  static SD_HOST_DEVICE bool togglePropertyBitForFlags(LongType &flagStorage, int property);
  static SD_HOST_DEVICE void unsetPropertyBitForFlags(LongType &flagStorage, int property);
  static SD_HOST_DEVICE SparseType sparseTypeForFlags(const LongType &flagStorage);
  static void setPropertyBitForFlagsValue(LongType &extraStorage, int property);
};

}
#endif  // ND4J_ARRAY_OPTIONS_H :)
