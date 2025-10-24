/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "execution/Threads.h"
#include "helpers/OpTracker.h"

#include <exceptions/allocation_exception.h>
#include <fcntl.h>
#include <graph/GraphExecutioner.h>

#include <helpers/BlasHelper.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <loops/type_conversions.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/transforms.h>
#include <stdio.h>
#include <stdlib.h>
#include <types/float8.h>
#include <types/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>

#else
#include <helpers/mman.h>
#include <io.h>
#endif
#include <errno.h>
#include <ops/declarable/CustomOperations.h>
#include <sys/types.h>


bool experimentalSupport = false;

// OpaqueNDArray allocation tracking
static std::atomic<size_t> g_opaqueArrayCount{0};
static std::atomic<size_t> g_opaqueArrayBytes{0};
static std::mutex g_opaqueArrayMutex;

// InteropDataBuffer/OpaqueDataBuffer allocation tracking
static std::atomic<size_t> g_dataBufferCount{0};
static std::atomic<size_t> g_dataBufferBytes{0};
static std::mutex g_dataBufferMutex;

#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>

#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif
#include <array/DataType.h>
#include <array/DataTypeUtils.h>




/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, DataType srcType, Pointer hX, long N, DataType dstType, Pointer hZ);
 */
void convertTypes(sd::Pointer *extras, int srcTypeInt, sd::Pointer hX, sd::LongType N, int destType, sd::Pointer hZ) {
  sd::DataType srcType = sd::DataTypeUtils::fromInt(srcTypeInt);
  sd::DataType dstType = sd::DataTypeUtils::fromInt(destType);
  auto hx = reinterpret_cast<void *>(hX);
  auto hz = reinterpret_cast<void *>(hZ);

#ifdef HAS_FLOAT8
  if (srcType == sd::DataType::FLOAT8) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT8, float8, DOUBLE, double);
    } else
    #endif
    {
      sd_debug("Unsupported types conversion: [%s] -> [%s]\n",
               sd::DataTypeUtils::asString(srcType).c_str(),
               sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_FLOAT8

#ifdef HAS_INT8
  if (srcType == sd::DataType::INT8) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT8, int8_t, DOUBLE, double);
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_INT8

#ifdef HAS_UINT8
  if (srcType == sd::DataType::UINT8) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT8, UnsignedChar, DOUBLE, double);
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_UINT8

#ifdef HAS_FLOAT16
  if (srcType == sd::DataType::HALF) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), HALF, float16, DOUBLE, double);
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_FLOAT16

#ifdef HAS_INT16
  if (srcType == sd::DataType::INT16) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
       _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), INT16, int16_t, DOUBLE, double);
    } else
    #endif
    {
      printf("Unsupported types conversion: [%s] -> [%s]\n",
             sd::DataTypeUtils::asString(srcType).c_str(),
             sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_INT16

#ifdef HAS_FLOAT32
  if (srcType == sd::DataType::FLOAT32) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      // No conversion needed - same type
      if (hx != hz) {
        _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, FLOAT32, float);
      }
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), FLOAT32, float, DOUBLE, double);
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_FLOAT32

#ifdef HAS_DOUBLE
  if (srcType == sd::DataType::DOUBLE) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      // No conversion needed - same type
      if (hx != hz) {
        _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), DOUBLE, double, DOUBLE, double);
      }
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_DOUBLE

#ifdef HAS_BFLOAT16
  if (srcType == sd::DataType::BFLOAT16) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, UINT16, uint16_t);
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, DOUBLE, double);
    } else
    #endif
    #ifdef HAS_BFLOAT16
    if (dstType == sd::DataType::BFLOAT16) {
      // No conversion needed - same type
      if (hx != hz) {
        _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), BFLOAT16, bfloat16, BFLOAT16, bfloat16);
      }
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_BFLOAT16

#ifdef HAS_UINT16
  if (srcType == sd::DataType::UINT16) {
    #ifdef HAS_FLOAT8
    if (dstType == sd::DataType::FLOAT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, FLOAT8, float8);
    } else
    #endif
    #ifdef HAS_INT8
    if (dstType == sd::DataType::INT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, INT8, int8_t);
    } else
    #endif
    #ifdef HAS_UINT8
    if (dstType == sd::DataType::UINT8) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, UINT8, uint8_t);
    } else
    #endif
    #ifdef HAS_FLOAT16
    if (dstType == sd::DataType::HALF) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, HALF, float16);
    } else
    #endif
    #ifdef HAS_INT16
    if (dstType == sd::DataType::INT16) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, INT16, int16_t);
    } else
    #endif
    #ifdef HAS_UINT16
    if (dstType == sd::DataType::UINT16) {
      // No conversion needed - same type
      if (hx != hz) {
        _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, UINT16, uint16_t);
      }
    } else
    #endif
    #ifdef HAS_FLOAT32
    if (dstType == sd::DataType::FLOAT32) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, FLOAT32, float);
    } else
    #endif
    #ifdef HAS_DOUBLE
    if (dstType == sd::DataType::DOUBLE) {
      _CALL_DOUBLE2(sd::TypeCast::convertGeneric, (nullptr, hx, N, hz), UINT16, uint16_t, DOUBLE, double);
    } else
    #endif
    {
      sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
                sd::DataTypeUtils::asString(srcType).c_str(),
                sd::DataTypeUtils::asString(dstType).c_str());
    }
  } else
#endif // HAS_UINT16

  {
    sd_printf("Unsupported types conversion: [%s] -> [%s]\n",
              sd::DataTypeUtils::asString(srcType).c_str(),
              sd::DataTypeUtils::asString(dstType).c_str());
  }
}

static long lengthInBytes(OpaqueDataBuffer *buffer) {
  return buffer->dataBuffer()->getLenInBytes();
}

template <typename T>
static sd::Pointer _numpyHeaderForNd4j(sd::Pointer data, const sd::Pointer shapeBuffer, sd::LongType wordSize,
                                       sd::LongType* headerSize) {
  sd::LongType const* shapeBufferCast = reinterpret_cast<const sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  const sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  char* ret = new char[npHeader.size() + 1];
  int count = 0;
  for (size_t i = 0; i < npHeader.size(); i++) {
    ret[count] = npHeader[i];
    count++;
  }

  ret[count] = '\0';
  count++;

  *headerSize = count;
  return reinterpret_cast<sd::Pointer>(ret);
}


sd::Pointer numpyHeaderForNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize,
                               sd::LongType* headerSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderForNd4j, (data, shapeBuffer, wordSize, headerSize), SD_COMMON_TYPES);
  return nullptr;
}

/**
 * Load numpy from a header
 * based on the cnpy parse from header method.
 * @param data the header data to parse
 * @return a pointer to a numpy cnpy:NpyArray struct
 */
sd::Pointer loadNpyFromHeader(sd::Pointer data) {
  char* header = reinterpret_cast<char*>(data);

  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(header);
  cnpy::NpyArray* ret = new cnpy::NpyArray();

  ret->data = arr.data;
  ret->wordSize = arr.wordSize;
  ret->shape = arr.shape;
  return reinterpret_cast<sd::Pointer>(ret);
}


/**
 * Create a numpy array from an nd4j
 * array
 * @param data a pointer to the data
 * @param shapeBuffer  the shapebuffer for the nd4j array
 * @param wordSize  the word size (4 for float, 8 for doubles)
 * @return a pointer to a numpy array
 */

template <typename T>
sd::Pointer _numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>( npShape, rank, wordSize);
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  char* dataChar = reinterpret_cast<char*>(data);
  char* npHeaderData = npHeader.data();
  char* ret = new char[(wordSize * length) + npHeader.size()];
  char* cursorStart = ret + npHeader.size();
  std::memcpy(ret, npHeaderData,
              npHeader.size());
  std::memcpy(cursorStart, dataChar,length  * wordSize);
  sd::Pointer rettPointer = reinterpret_cast<sd::Pointer>(ret);
  return rettPointer;
}
template<typename T>
long _numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  sd::LongType wordSize = opaqueDataBuffer->dataBuffer()->getLenInBytes() / opaqueDataBuffer->dataBuffer()->getNumElements();
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  return ret;
}

template<typename  T>
long _numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  return ret;
}



long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLengthWordSize, (shapeBuffer, wordSize), SD_COMMON_TYPES);
  return 0;

}

long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLength, (opaqueDataBuffer, shapeBuffer), SD_COMMON_TYPES);
  return 0;

}



sd::Pointer numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyFromNd4j, (data, shapeBuffer, wordSize), SD_COMMON_TYPES);
  return nullptr;
}


sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray) {
  try {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int shapeSize = arr.shape.size();
    std::vector<sd::LongType> shape(shapeSize);
    bool _empty = false;
    for (unsigned int i = 0; i < shapeSize; i++) {
      shape[i] = arr.shape[i];

      if (arr.shape[i] == 0) _empty = true;
    }

    auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

    sd::LongType *shapeBuffer;
    if (shape.size() == 1 && shape[0] == 0) {
      // scalar case
      shapeBuffer = sd::ShapeBuilders::createScalarShapeInfo(dtype);
    } else if (_empty) {
      if (shapeSize > 0)
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
      else
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype);
    } else {
      shapeBuffer = sd::ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
    }
    return (sd::Pointer)(sd::ConstantShapeHelper::getInstance().createFromExisting(
        shapeBuffer));  // TO DO: this can lead to unpleasant crash sometimes
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueNDArray getOutputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->outputArray(idx);
}


OpaqueNDArray getInputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->array(idx);
}


sd::LongType dataTypeNativeAt(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return static_cast<sd::LongType>(ptr->dataType(idx));

}


bool bArgAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return false;
  return ptr->getBArguments()->at(idx);

}

sd::LongType iArgumentAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return ptr->getIArguments()->at(idx);

}

sd::LongType numDNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numD();
}

sd::LongType numBNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numB();
}

sd::LongType numOutputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->outputWidth();
}
sd::LongType numInputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->width();
}

double tArgumentNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0.0;
  return ptr->getTArguments()->at(idx);
}

sd::LongType numTArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numT();
}

sd::LongType numIArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numI();
}




void setGraphContextOutputArray(OpaqueContext* ptr, int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextOutputArray: Input arrays were null!");

  ptr->setOutputArray(index,arr,false);


}

void setGraphContextInputArray(OpaqueContext* ptr,int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArray: Input arrays were null!");

  ptr->setInputArray(index, arr, false);

}

//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if (arr == nullptr) THROW_EXCEPTION("setGraphContextOutputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if (arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextOutputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    for (int j = 0; j < numArrays; j++) {
      ptr->setOutputArray(j, *arr[j], false);
    }
  }
}


sd::LongType getOpaqueNDArrayLeakCount() {
  return static_cast<sd::LongType>(g_opaqueArrayCount.load(std::memory_order_relaxed));
}

sd::LongType getOpaqueNDArrayLeakBytes() {
  return static_cast<sd::LongType>(g_opaqueArrayBytes.load(std::memory_order_relaxed));
}



sd::Pointer createUtf8String(sd::Pointer *extraPointers, const char *string, int length) {
  auto u = new sd::utf8string(string, length);
  return reinterpret_cast<sd::Pointer>(u);
}

sd::LongType getUtf8StringLength(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_length;
}
char *getUtf8StringBuffer(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_buffer;
}

void deleteUtf8String(sd::Pointer *extraPointers, sd::Pointer ptr) { delete (reinterpret_cast<sd::utf8string *>(ptr)); }

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }



OpaqueConstantShapeBuffer shapeBufferEx(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                        char order,
                                        sd::LongType ews, sd::LongType extras) {

    auto desc = sd::ShapeBuilders::createShapeInfo(dtype, order,rank, shape, strides,nullptr, extras);
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    delete[] desc;
    return buffer;

}

void inspectArray(sd::Pointer *extraPointers, sd::Pointer buffer, sd::LongType *shapeInfo, sd::Pointer specialBuffer,
                  sd::LongType *specialShapeInfo, sd::Pointer debugInfo) {
  try {
    auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
    sd::NDArray array(buffer, shapeInfo, nullptr, 0, 0);
    sd::DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }


}



void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) { }

void deleteConstantDataBuffer(OpaqueConstantDataBuffer *ptr) {
  delete ptr;
}

OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(sd::LongType *shapeInfo) {
  try {
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());

  }

  return nullptr;
}

sd::LongType *mmapFile(sd::Pointer *extraPointers, const char *fileName, sd::LongType length) {
  auto hZ = new sd::LongType[2];
  sd::LongType ptr = 0;
  errno = 0;
  try {
#if defined(_WIN32) || defined(_WIN64)
    _mmap(hZ, static_cast<size_t>(length), fileName);
    _mmap(hZ, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);  // checking for failed fopen
    if (fd < 0) {
      sd_printf("Errno: %i\n", errno);
      THROW_EXCEPTION("Failed to open file for MMAP");
    }

    void *ptr2 = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0);
    if (ptr2 == MAP_FAILED) {
      sd_printf("Errno: %i\n", errno);
      THROW_EXCEPTION("Failed to mmap file");
    }
    hZ[0] = (sd::LongType)ptr2;
    hZ[1] = fd;

#endif

    return hZ;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }

  return nullptr;
}
void munmapFile(sd::Pointer *extraPointers, sd::LongType  *ptrMap, sd::LongType  length) {}

ResultWrapper *executeFlatGraph(sd::Pointer *extraPointers, sd::Pointer flatBufferPointer) {
  try {
    return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType  getResultWrapperSize(ResultWrapper *ptr) { return ptr->size(); }
sd::Pointer getResultWrapperPointer(ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return sd::ops::OpRegistrator::getInstance().getAllCustomOperations(); }

OpaqueShapeList *calculateOutputShapes2(sd::Pointer *extraPointers, sd::LongType hash, OpaqueContext *context) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    sd::ShapeList inShapes;

    for (size_t e = 0; e < context->width(); e++) {
      if (context->array(e) == nullptr) {
        std::string errorMessage = "Input array at index " + std::to_string(e) + " was null!";
        THROW_EXCEPTION(errorMessage.c_str());
      }
      inShapes.push_back(context->array(e)->shapeInfo());
    }

    auto shapeList = op->calculateOutputShape(&inShapes, *context);
    return shapeList;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

bool checkOpaqueNDArrayElementsNull(OpaqueNDArrayArr elements,int numElements) {
  for (int i = 0; i < numElements; i++) {
    if (elements[i] == nullptr) return true;
  }
  return false;
}


sd::LongType  getShapeListSize(sd::ShapeList *list) { return list->size(); }

sd::LongType  const *getShape(sd::ShapeList *list, sd::LongType  i) { return list->at(i); }




// Function to execute a custom operation
sd::Status execCustomOp(sd::Pointer *extraPointers, sd::LongType  hash, OpaqueNDArrayArr inputs, int numInputs,
                        OpaqueNDArrayArr outputs, int numOutputs, double *tArgs, int numTArgs,
                        sd::LongType  *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
  try {
    // Convert NDArray** inputs and outputs to std::vector<NDArray*>
    const std::vector<sd::NDArray*> inputVec(inputs, inputs + numInputs);
    const std::vector<sd::NDArray*> outputVec(outputs, outputs + numOutputs);
    const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
    const std::vector<sd::LongType > iArgsVec(iArgs, iArgs + numIArgs);
    const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

    // Retrieve the operation based on the hash
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) {
      throw std::invalid_argument("Operation not found for the given hash.");
    }

    // Execute the custom operation
    return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::KERNEL_FAILURE;
  }
}

void toggleOpTrace(bool opTrace) { sd::ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() { sd::ops::OpRegistrator::getInstance().purgeOpExecs();
}




void printOpTrace() {
  auto execTrace = *sd::ops::OpRegistrator::getInstance().execTrace();
  for(size_t i = 0; i < execTrace.size(); i++) {
    auto curr = execTrace[i];
    if(curr->opName != nullptr) {
      sd_printf("Op name: %s\n", curr->opName->c_str());
    }
    sd_printf(" Input buffers:\n",0);
    if(curr->inputShapeBuffers == nullptr || curr->inputShapeBuffers->size() == 0) {
      sd_printf("No input buffers\n",0);
      continue;
    } else {
      auto currInputShapeBuffers = *(curr->inputShapeBuffers);
      for(size_t j = 0; j < currInputShapeBuffers.size(); j++) {
        auto buff = currInputShapeBuffers[j];
        shape::printShapeInfo(buff);
        sd_printf("\n",0);
      }
    }

    if(curr->outputShapeBuffers == nullptr || curr->outputShapeBuffers->size() == 0) {
      sd_printf("No output buffers\n",0);
      continue;
    } else {
      auto currOutputShapeBuffers = *(curr->outputShapeBuffers);
      for(size_t j = 0; j < curr->outputShapeBuffers->size(); j++) {
        shape::printShapeInfo(currOutputShapeBuffers[j]);
        sd_printf("\n",0);
      }

    }


  }

}


std::vector<ExecTrace*> * listOpTraces() {
  return sd::ops::OpRegistrator::getInstance().execTrace();
}

void deleteNDArray(OpaqueNDArray array) {
  if (array == nullptr) {
    return;
  }

  // Track deallocation
  size_t bytes = array->lengthOf() * array->sizeOfT();
  g_opaqueArrayCount.fetch_sub(1, std::memory_order_relaxed);
  g_opaqueArrayBytes.fetch_sub(bytes, std::memory_order_relaxed);

  if(sd::Environment::getInstance().isVerbose()) {
    sd_printf("deleteNDArray: deallocating array at %p, count=%zu, total_bytes=%zu, freed_bytes=%zu\n",
              array, g_opaqueArrayCount.load(), g_opaqueArrayBytes.load(), bytes);
  }

  delete array;
}

sd::LongType getOpaqueNDArrayOffset(OpaqueNDArray array) {
  return array->offset();
}


const sd::LongType* getOpaqueNDArrayShapeInfo(OpaqueNDArray array) {
  return array->shapeInfo();
}



void* getOpaqueNDArrayBuffer(OpaqueNDArray array) {
  if(array == nullptr || array->dataBuffer() == nullptr) {
    THROW_EXCEPTION("getOpaqueNDArrayBuffer: Array or data buffer was null!");
  }
  return array->dataBuffer()->primary();
}

void* getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) {
  if(array == nullptr || array->dataBuffer() == nullptr) {
    THROW_EXCEPTION("getOpaqueNDArraySpecialBuffer: Array or data buffer was null!");
  }
  return array->dataBuffer()->special();
}

sd::LongType getShapeInfoLength(OpaqueNDArray array) {
  return shape::shapeInfoLength(array->rankOf());
}

sd::LongType getOpaqueNDArrayLength(OpaqueNDArray array) {
  return array->dataBuffer()->getNumElements();
}


OpaqueNDArray createOpaqueNDArray(OpaqueDataBuffer *shapeInfo,
                                  OpaqueDataBuffer *buffer,
                                  OpaqueDataBuffer *specialBuffer,
                                  sd::LongType offset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was null!");
  }

  sd::LongType* shapeInfoCast = reinterpret_cast<sd::LongType*>(shapeInfo->primary());

  if(shape::isEmpty(shapeInfoCast) && buffer != nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was empty but buffer was not null!");
  } else if(!shape::isEmpty(shapeInfoCast) && buffer == nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was not empty but buffer was null!");
  }

  sd::NDArray* ret = new sd::NDArray(
    buffer != nullptr ? buffer->getDataBuffer() : nullptr,
    shapeInfoCast,
    sd::LaunchContext::defaultContext(),
    offset
  );

  // Track allocation
  if (ret != nullptr) {
    size_t bytes = ret->lengthOf() * ret->sizeOfT();
    g_opaqueArrayCount.fetch_add(1, std::memory_order_relaxed);
    g_opaqueArrayBytes.fetch_add(bytes, std::memory_order_relaxed);

    if(sd::Environment::getInstance().isVerbose()) {
      sd_printf("createOpaqueNDArray: allocated array at %p, count=%zu, total_bytes=%zu, this_bytes=%zu\n",
                ret, g_opaqueArrayCount.load(), g_opaqueArrayBytes.load(), bytes);
    }
  }

  return ret;
}


void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  sd::DataBuffer::memcpy(target->dataBuffer(), from->dataBuffer(), targetOffset, fromOffset);
}



int contextNumInputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->width();
}

int contextNumOutputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->outputWidth();
}



int numInputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers->size();
}

int numOutputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers->size();
}

std::vector<bool> * bArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &trace->bArgs;
}

std::vector<std::string> * sArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->sArguments);
}
std::vector<double> * tArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->tArgs);

}

std::vector<int> * dArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  std::vector<int> *dArgs = new std::vector<int>();
  for (size_t e = 0; e < trace->dArgs.size(); e++) {
    dArgs->push_back(trace->dArgs[e]);
  }
  return dArgs;
}

std::vector<sd::LongType> * iArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &(trace->iArgs);
}

std::vector<const sd::LongType *> *inputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers;
}

std::vector<const sd::LongType *> *outputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers;
}

char *opName(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return const_cast<char *>(trace->opName->c_str());
}

void setElementThreshold(int num) {
  if (num > 0) sd::Environment::getInstance().setElementwiseThreshold(num);
}

void setTADThreshold(int num) {
  if (num > 0) sd::Environment::getInstance().setTadThreshold(num);
}


sd::Status registerGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer flatBufferPointer) {
  try {
    auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    GraphHolder::getInstance().registerGraph(graphId, graph);

    return sd::Status::OK;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

static VariablesSet *executeStoredGraphT(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers,
                                         sd::Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = sd::graph::GraphHolder::getInstance().cloneGraph(graphId);
  auto varSpace = graph->getVariableSpace();

  std::vector<sd::NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new sd::NDArray(inputBuffers[e], reinterpret_cast<sd::LongType  *>(inputShapes[e]), nullptr, 0, 0);
    handles.emplace_back(array);

    if (varSpace->hasVariable(idx)) {
      auto var = varSpace->getVariable(idx);
      if (var->hasNDArray()) delete var->getNDArray();

      var->setNDArray(array);
    } else
      varSpace->putVariable(idx, array);
  }

  auto hZ = sd::graph::GraphExecutioner::execute(graph, varSpace);
  auto varSet = new sd::graph::VariablesSet(hZ);

  if (hZ == sd::Status::OK) {
    // pull back results, and provide them
    auto outputs = graph->fetchOutputs();
    int size = static_cast<int>(outputs->size());
    for (int e = 0; e < size; e++) {
      // we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
      std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

      auto var = varSpace->getVariable(varId);

      varSet->push_back(var->clone());
    }

    delete outputs;
  }

  delete graph;

  return varSet;
}


VariablesSet *executeStoredGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers, sd::Pointer *inputShapes,
                                 int *inputIndices, int numInputs) {
  try {
    return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType  getVariablesSetSize(OpaqueVariablesSet *set) { return set->size(); }

sd::Status getVariablesSetStatus(OpaqueVariablesSet *set) { return set->status(); }

OpaqueVariable *getVariable(OpaqueVariablesSet *set, sd::LongType  i) { return set->at(i); }

int getVariableId(Variable *variable) { return variable->id(); }

int getVariableIndex(Variable *variable) { return variable->index(); }

const char *getVariableName(Variable *variable) { return variable->getName()->c_str(); }

sd::LongType  const *getVariableShape(Variable *variable) { return variable->getNDArray()->shapeInfo(); }

void *getVariableBuffer(Variable *variable) { return variable->getNDArray()->buffer(); }

sd::Status unregisterGraph(sd::Pointer *extraPointers, sd::LongType  graphId) {
  try {
    GraphHolder::getInstance().dropGraphAny(graphId);

    return sd::Status::OK;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

void deletePointerArray(sd::Pointer pointer) {
  sd::Pointer *ptr = reinterpret_cast<sd::Pointer *>(pointer);
  delete[] ptr;
}

void deleteCharArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<char *>(pointer);
  delete[] ptr;
}

void deleteIntArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<int *>(pointer);
  delete[] ptr;
}

void deleteLongArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<sd::LongType  *>(pointer);
  delete[] ptr;
}

void deleteVariablesSet(VariablesSet *pointer) {
  delete pointer;
}

void deleteShapeList(sd::Pointer shapeList) {
  sd::ShapeList *list = reinterpret_cast<sd::ShapeList *>(shapeList);
  delete list;
}

const char *getAllOperations() { return sd::OpTracker::getInstance().exportOperations(); }

sd::Pointer getGraphState(sd::LongType  id) { return (sd::Pointer) new GraphState(id); }

void deleteGraphState(sd::Pointer state) {
  auto stateP = reinterpret_cast<GraphState *>(state);
  delete stateP;
}

sd::Status execCustomOpWithScope_(sd::Pointer *extraPointers, sd::graph::GraphState *state, sd::LongType  opHash,
                                  sd::LongType  *scopes, int numScopes, sd::Pointer *inputBuffers,
                                  sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers,
                                  sd::Pointer *outputShapes, int numOutputs) {
  /**
   * That's basically exec, with VariableSpace provided in GraphState:
   * depending on operation (i.e. while of if), different logic executors could be used
   */

  auto graph = state->graph();
  auto varSpace = state->variableSpace();

  // Node is dynamically created, and has nothing beyond it: only inputs and outputs
  // this node has id of 0, and inputs are
  Node node(::graph::OpType_LOGIC, opHash, 0);

  // mapping inputs
  for (int e = 0; e < numInputs; e++) {
    auto buffer = inputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(inputShapes[e]);

    auto array = new sd::NDArray(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace
    varSpace->putVariable(0, e, *array);
    node.pickInput(0, e);
  }

  // mapping scopes
  for (int e = 0; e < numScopes; e++) {
    // we should check scope existence in GraphState/Graph
    int scopeId = (int)scopes[e];
    if (!state->hasScope(scopeId)) {
      return sd::Logger::logKernelFailureMsg();
    }
    node.pickInput(scopeId, 0);
  }

  auto hZ = LogicExecutor::processNode(graph, &node);
  if (hZ != sd::Status::OK) return hZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(outputShapes[e]);

    sd::NDArray array(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace to the same ID
    // varSpace->putVariable(0, e, array);

    auto t = varSpace->getVariable(0, e)->getNDArray();
    array.assign(t);
  }

  // removing input variables
  for (int e = 0; e < numInputs; e++) {
    varSpace->dropVariable(0, e);
  }

  return sd::Status::OK;
}

void deleteResultWrapper(sd::Pointer ptr) {
  auto p = reinterpret_cast<ResultWrapper *>(ptr);
  delete p;
}


template <typename T>
SD_INLINE int estimateThresholdGeneric(sd::Pointer *extraPointers, sd::Pointer hX, int N, float threshold) {
  auto buffer = reinterpret_cast<T *>(hX);
  int span = (N / 6) + 8;
  // Cast the threshold to the appropriate type T
  T typedThreshold = static_cast<T>(threshold);

  auto func = PRAGMA_REDUCE_LONG {
    int64_t cnt = 0;
    PRAGMA_OMP_SIMD
    for (auto e = start; e < stop; e++) {
      auto v = sd::math::sd_abs<T,T>(buffer[e]);
      if (v >= typedThreshold) cnt++;
    }

    return cnt;
  };

  return samediff::Threads::parallel_long(
      func, LAMBDA_AL { return _old + _new; }, 0, N);
}

int estimateThreshold(sd::Pointer *extraPointers, sd::Pointer hX, sd::LongType const *hXShapeInfo, int N,
                      float threshold) {
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), SD_FLOAT_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return 0;
  }

  return 0;
}



void deleteTadPack(sd::TadPack *ptr) {
  delete ptr;
}




OpaqueConstantDataBuffer constantBufferLong(sd::DataType dtype, sd::LongType  *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBufferDouble(sd::DataType dtype, double *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor) {
  return sd::ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
}

sd::Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf) { return dbf->primary(); }
sd::Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf) { return dbf->special(); }
sd::LongType getConstantDataBufferLength(OpaqueConstantDataBuffer dbf) { return dbf->length(); }
sd::LongType getConstantDataBufferSizeOf(OpaqueConstantDataBuffer dbf) { return dbf->sizeOf(); }

sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->primary()); }

sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->special()); }

Context *createGraphContext(int nodeId) { return new Context(nodeId); }

OpaqueRandomGenerator getGraphContextRandomGenerator(Context *ptr) { return &ptr->randomGenerator(); }

void markGraphContextInplace(Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }


//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextInputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if(arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextInputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    OpaqueNDArray &ref = *arr[i];
    ptr->setInputArray(i, ref, false);
  }
}



void setGraphContextTArguments(Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}

void setGraphContextIArguments(Context *ptr, sd::LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}

void setGraphContextBArguments(Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<sd::DataType> dtypes(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) dtypes[e] = sd::DataTypeUtils::fromInt(arguments[e]);

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(Context *ptr) {
  delete ptr;
}

OpaqueRandomGenerator createRandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed) {
  try {
    return new RandomGenerator(rootSeed, nodeSeed);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType getRandomGeneratorRootState(OpaqueRandomGenerator ptr) { return ptr->rootState(); }

sd::LongType getRandomGeneratorNodeState(OpaqueRandomGenerator ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(OpaqueRandomGenerator ptr, sd::LongType rootSeed, sd::LongType nodeSeed) {
  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, sd::LongType index) { return ptr->relativeInt(index); }

sd::LongType getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

sd::LongType getRandomGeneratorNextLong(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(OpaqueRandomGenerator ptr) { delete ptr; }


/**
 * Get the shape buffer from a
 * numpy array.
 * **Warning** this allocates memory
 * @param npyArray
 * @return
 */
sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  auto shape = new sd::LongType[arr.shape.size()];
  for (unsigned int i = 0; i < arr.shape.size(); i++) {
    shape[i] = arr.shape[i];
  }

  auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
  delete[] shape;
  return reinterpret_cast<sd::Pointer>(shapeBuffer);
}

/**
 *
 * @param npyArray
 * @return
 */
sd::Pointer dataPointForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arr.data);
  return dataToPrint;
}

/**
 *
 * @param npyArray
 * @return
 */
sd::Pointer dataPointForNumpyStruct(sd::Pointer npyArrayStruct) {
  cnpy::NpyArray* arrPointer = reinterpret_cast<cnpy::NpyArray*>(npyArrayStruct);
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arrPointer->data);
  return reinterpret_cast<sd::Pointer>(dataToPrint);
}

/**
 *
 * @param npyArray
 * @param fromFile
 * @return
 */
sd::Pointer dataPointForNumpy(sd::Pointer npyArray) {
  char* npyArrayBuffer = reinterpret_cast<char*>(npyArray);
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(npyArrayBuffer);
  return dataPointForNumpyStruct(reinterpret_cast<sd::Pointer>(&arr));
}

/**
 * Load a numpy array from a file
 * and return it as an sd::Pointer
 * @param path
 * @return
 */
sd::Pointer numpyFromFile(std::string path) {
  char* numpyBuffer = cnpy::loadFile(path.data());
  return reinterpret_cast<sd::Pointer>(numpyBuffer);
}

////// NPZ //////

void* mapFromNpzFile(std::string path) {
  cnpy::npz_t* mapPtr = new cnpy::npz_t();
  cnpy::npz_t map = cnpy::npzLoad(path);
  mapPtr->insert(map.begin(), map.end());
  return reinterpret_cast<void*>(mapPtr);
}

int getNumNpyArraysInMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  int n = arrays->size();
  return n;
}

const char* getNpyArrayNameFromMap(void* map, int index, char* nameBuffer) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      size_t len_of_str = strlen(it->first.c_str());
      memcpy(nameBuffer, it->first.c_str(), len_of_str);
    }
  }
  return "";
}

void* getNpyArrayFromMap(void* map, int index) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  cnpy::NpyArray* arr = new cnpy::NpyArray();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      *arr = it->second;
      return arr;
    }
  }

 return nullptr;
}


void* getNpyArrayData(void* npArray) {
  cnpy::NpyArray* npyArray2 = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return reinterpret_cast<void*>(npyArray2->data);
}

int getNpyArrayRank(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int rank = arr->shape.size();
  return rank;
}

sd::LongType* getNpyArrayShape(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int ndim = arr->shape.size();
  sd::LongType* shape = new sd::LongType[ndim];
  for (int i = 0; i < ndim; i++) {
    shape[i] = arr->shape.at(i);
  }
  return shape;
}

char getNpyArrayOrder(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return (arr->fortranOrder) ? 'f' : 'c';
}

int getNpyArrayElemSize(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return arr->wordSize;
}

void deleteNPArrayStruct(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  delete arr;
}

void deleteNPArrayMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  delete arrays;
}
//////

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArray(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  // arrPointer->destruct();
  return size;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArrayHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

void releaseNumpy(sd::Pointer npyArray) { free(reinterpret_cast<void*>(npyArray)); }

#if defined(SD_GCC_FUNCTRACE)
// this is mainly a c based function.
extern "C" {

//note this is a c++ 17 feature
#ifndef INSTRUMENT_FILE_DEF
#define INSTRUMENT_FILE_DEF 1
FILE* instrumentFile = nullptr;
#endif





}

#endif

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

sd::LongType getCachedMemory(int deviceId) { return sd::ConstantHelper::getInstance().getCachedAmount(deviceId); }


void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) {
  ptr->setShapeFunctionOverride(reallyOverride);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

int lastErrorCode() { return sd::LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return sd::LaunchContext::defaultContext()->errorReference()->errorMessage(); }


sd::LaunchContext *defaultLaunchContext() { return sd::LaunchContext::defaultContext(); }





void setIntermediateResult(OpaqueContext *contextPointer,
                           int index,
                           OpaqueDataBuffer *buffer,
                           OpaqueDataBuffer *shapeInfo,
                           sd::LongType dataOffset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("Set Intermediate Result: shapeInfo is null");
  }
  auto casted = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(casted, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(),
                             desc,
                             sd::LaunchContext::defaultContext(),
                             dataOffset);
  contextPointer->setIntermediateResult(index, arr);
}


std::vector<const sd::LongType *> intermediateResultsShapeInfo(OpaqueContext *contextPointer) {
  std::vector<const sd::LongType *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    const sd::LongType *buff = v->shapeInfo();
    intermediates.push_back(buff);
  }

  return intermediates;
}

std::vector<OpaqueDataBuffer *> intermediateResults(OpaqueContext *contextPointer) {
  std::vector<OpaqueDataBuffer *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    OpaqueDataBuffer *buff = new OpaqueDataBuffer (v->dataBuffer());
    intermediates.push_back(buff);
  }

  return intermediates;
}

int numIntermediateResults(OpaqueContext *contextPointer) {
  return contextPointer->numIntermediates();
}

void pushIntermediateResult(OpaqueContext *contextPointer,
                            OpaqueDataBuffer *buffer,
                            OpaqueDataBuffer *shapeInfo,
                            sd::LongType offset) {
  auto shapeInfoCast = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(shapeInfoCast, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(), desc, sd::LaunchContext::defaultContext(), offset);
  contextPointer->pushIntermediateResult(arr);
}

OpaqueDataBuffer  * intermediateResultDataAt(int index, OpaqueContext *contextPointer) {
  auto arr = contextPointer->intermediateResult(index);
  return new OpaqueDataBuffer(arr->dataBuffer());
}

const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer) {
  auto context = reinterpret_cast<sd::graph::Context *>(contextPointer);
  auto arr = context->intermediateResult(index);
  return arr->shapeInfo();
}


sd::LongType const *getPrimaryShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->primaryShapeInfo());
}

sd::LongType const *getPrimaryOffsets(sd::TadPack *pack) {
  if(pack->primaryOffsets() == nullptr)
    THROW_EXCEPTION("getPrimaryOffsets: primaryOffsets is nullptr!");
  return const_cast<sd::LongType *>(pack->primaryOffsets());
}

sd::LongType const *getSpecialShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->specialShapeInfo());
}

sd::LongType const *getSpecialOffsets(sd::TadPack *pack) { return const_cast<sd::LongType *>(pack->specialOffsets()); }

sd::LongType getNumberOfTads(sd::TadPack *pack) { return pack->numberOfTads(); }

int getShapeInfoLength(sd::TadPack *pack) { return pack->shapeInfoLength(); }


sd::TadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength) {
  try {
    if(hXShapeInfo->primary() == nullptr) {
      THROW_EXCEPTION("tadOnlyShapeInfo: hXShapeInfo->primary() is nullptr!");
    }

    auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());
    auto shapeFromCache = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(buffPrim)->primary();
    auto rankVal = shapeFromCache[0];
    if(rankVal == 0) {
      //detect when the shape buffer values are unset.
      auto len = shape::shapeInfoLength(rankVal);
      //min number of values in a shape info buffer
      bool allZero = true;
      for(int i = 0; i < len; i++) {
        if(buffPrim[i] != 0) {
          allZero = false;
          break;
        }
      }

      if(allZero) {
        THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
      }
    }




    auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(
        shapeFromCache, dimension, dimensionLength);
    return pack;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }

  return nullptr;

}


OpaqueConstantShapeBuffer shapeBuffer(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                      char order, sd::LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

void dbPrintAllocationTrace(OpaqueDataBuffer *db) { db->dataBuffer()->printAllocationTrace(); }

sd::LongType dbBufferLength(OpaqueDataBuffer *dataBuffer) {
  return dataBuffer->dataBuffer()->getNumElements();
}


OpaqueDataBuffer *dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = sd::DataTypeUtils::fromInt(dataType);
    sd::LongType totalElementSize = elements == 0 ? sd::DataTypeUtils::sizeOf(dtype) : elements * sd::DataTypeUtils::sizeOf(dtype);
    auto buffer = new sd::InteropDataBuffer(totalElementSize, dtype, allocateBoth);

    // Track allocation
    if (buffer != nullptr) {
      size_t bytes = totalElementSize;
      g_dataBufferCount.fetch_add(1, std::memory_order_relaxed);
      g_dataBufferBytes.fetch_add(bytes, std::memory_order_relaxed);

      if(sd::Environment::getInstance().isVerbose()) {
        sd_printf("allocateDataBuffer: allocated buffer at %p, count=%zu, total_bytes=%zu, this_bytes=%zu\n",
                  buffer, g_dataBufferCount.load(), g_dataBufferBytes.load(), bytes);
      }
    }

    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);
  buffer->markOwner(false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  return dataBuffer->primary();
}

sd::Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSpecialBuffer: dataBuffer is null");
  return dataBuffer->special();
}

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("deleteDataBuffer: dataBuffer is null");

  // Close the buffer first to ensure proper cleanup of underlying DataBuffer
  // This updates tracking counters and frees the actual data
  dbClose(dataBuffer);

  // Now delete the wrapper
  delete dataBuffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetPrimaryBuffer: dataBuffer is null");
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetSpecialBuffer: dataBuffer is null");
  dataBuffer->setSpecial(specialBuffer, numBytes);
}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocatePrimaryBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocatePrimary();
}

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocateSpecialBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocateSpecial();
}

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  try {
    if(dataBuffer == nullptr)
      THROW_EXCEPTION("dbExpandBuffer: dataBuffer is null");
    dataBuffer->dataBuffer()->expand(elements * sd::DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, sd::LongType length) {
  return new OpaqueDataBuffer(dataBuffer, length);
}


int dbUseCount(OpaqueDataBuffer* dataBuffer) {
  if(dataBuffer) return dataBuffer->useCount();
  return 0;
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToSpecial: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToSpecial();
}

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToPrimary: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(sd::LaunchContext::defaultContext(),false);

}

void dbTickHostRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readPrimary();
}

void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writePrimary();
}

void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readSpecial();
}

void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writeSpecial();

}

void dbExpand(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbExpand: dataBuffer is null");
  dataBuffer->expand(elements);
}

void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  // Check if already closed - this flag is in InteropDataBuffer, not the freed DataBuffer
  if(dataBuffer->_closed) {
    return;
  }

  // Check constant flag (public field, safe to access)
  if(dataBuffer->isConstant) {
    return;
  }

  // Check if we even have a DataBuffer pointer
  if(!dataBuffer->hasValidDataBuffer()) {
    dataBuffer->_closed = true;
    return;
  }

  // If we don't own it, don't close it
  if(!dataBuffer->isOwner()) {
    return;
  }

  // Track deallocation using cached size - DO NOT touch the DataBuffer as it may be freed
  // Use the cached size from InteropDataBuffer instead of accessing potentially freed memory
  size_t bytes = dataBuffer->_cachedLenInBytes;
  g_dataBufferCount.fetch_sub(1, std::memory_order_relaxed);
  g_dataBufferBytes.fetch_sub(bytes, std::memory_order_relaxed);

  if(sd::Environment::getInstance().isVerbose()) {
    sd_printf("dbClose: deallocating buffer at %p, count=%zu, total_bytes=%zu, freed_bytes=%zu\n",
              dataBuffer, g_dataBufferCount.load(), g_dataBufferBytes.load(), bytes);
  }

  // Do NOT call db->close() - if this InteropDataBuffer wraps a DataBuffer that was owned
  // by an NDArray, that DataBuffer may already be freed. The NDArray destructor will have
  // already called close() on its DataBuffer. Calling it again causes use-after-free.
  // Just update tracking and mark as closed.

  // Mark as closed and invalidate pointer after freeing
  dataBuffer->_closed = true;
  dataBuffer->invalidateDataBuffer();
}

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbDeviceId: dataBuffer is null");
  return dataBuffer->deviceId();
}

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetDeviceId: dataBuffer is null");
  dataBuffer->setDeviceId(deviceId);
}

int dbLocality(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbLocality: dataBuffer is null");
  auto p = dataBuffer->dataBuffer()->isPrimaryActual();
  auto d = dataBuffer->dataBuffer()->isSpecialActual();

  if (p && d)
    return 0;
  else if (p)
    return -1;
  else
    return 1;
}

