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


extern bool experimentalSupport; // Defined in NativeOpsHelpers_Arrays.cpp

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

