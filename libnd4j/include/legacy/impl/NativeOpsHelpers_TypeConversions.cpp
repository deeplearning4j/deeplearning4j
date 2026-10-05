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

// On Windows, include windows.h early so _WINDOWS_ is defined before types.h
// constexpr alias guards are evaluated (avoids BOOL/INT64/etc. typedef conflicts)
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "execution/Threads.h"
#include "helpers/OpTracker.h"


#include <fcntl.h>

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
#include <sys/types.h>


extern bool experimentalSupport; // Defined in NativeOpsHelpers_Arrays.cpp

// External references to allocation tracking variables (defined in NativeOpsHelpers_Arrays.cpp and NativeOpsHelpers_DataBuffers.cpp)
extern std::atomic<size_t> g_opaqueArrayCount;
extern std::atomic<size_t> g_opaqueArrayBytes;
extern std::mutex g_opaqueArrayMutex;

extern std::atomic<size_t> g_dataBufferCount;
extern std::atomic<size_t> g_dataBufferBytes;
extern std::mutex g_dataBufferMutex;


#include <execution/Threads.h>
#include <graph/Context.h>
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

namespace {

// The element codes convertTypes takes are NDArrayFactory.convertDataEx's on every backend: the ordinals of
// org.nd4j.linalg.api.buffer.DataTypeEx. They are not sd::DataType values: DataTypeEx.DOUBLE is 7, which as a DataType
// is INT8, so reading the codes as DataTypes copied one byte per element of a DOUBLE-to-DOUBLE conversion.
enum ConversionCode : int {
  CONVERSION_FLOAT8 = 0,
  CONVERSION_INT8 = 1,
  CONVERSION_UINT8 = 2,
  CONVERSION_FLOAT16 = 3,
  CONVERSION_INT16 = 4,
  CONVERSION_UINT16 = 5,
  CONVERSION_FLOAT32 = 6,
  CONVERSION_DOUBLE = 7,
  CONVERSION_THRESHOLD = 8,
  CONVERSION_FLEXIBLE_THRESHOLD = 9
};

// The element type a code stands for; UNKNOWN for the threshold encodings and for any other code.
sd::DataType conversionElementType(int code) {
  switch (code) {
    case CONVERSION_FLOAT8:
      return sd::DataType::FLOAT8;
    case CONVERSION_INT8:
      return sd::DataType::INT8;
    case CONVERSION_UINT8:
      return sd::DataType::UINT8;
    case CONVERSION_FLOAT16:
      return sd::DataType::HALF;
    case CONVERSION_INT16:
      return sd::DataType::INT16;
    case CONVERSION_UINT16:
      return sd::DataType::UINT16;
    case CONVERSION_FLOAT32:
      return sd::DataType::FLOAT32;
    case CONVERSION_DOUBLE:
      return sd::DataType::DOUBLE;
    default:
      return sd::DataType::UNKNOWN;
  }
}

std::string conversionName(int code) {
  if (code == CONVERSION_THRESHOLD) return "THRESHOLD";
  if (code == CONVERSION_FLEXIBLE_THRESHOLD) return "FTHRESHOLD";
  const sd::DataType type = conversionElementType(code);
  return type == sd::DataType::UNKNOWN ? "code " + std::to_string(code) : sd::DataTypeUtils::asString(type);
}

void rejectConversion(int srcCode, int dstCode, const char *reason) {
  const std::string message =
      "convertTypes: " + conversionName(srcCode) + " -> " + conversionName(dstCode) + ": " + reason;
  THROW_EXCEPTION(message.c_str());
}

// THRESHOLD as the destination encodes a dense FLOAT16, FLOAT or DOUBLE array into the int encoding of the THRESHOLD
// compression codec, taking the encoded updates out of the dense array; as the source it adds an encoding's updates
// into a dense array. Both are host loops (loops/impl/type_conversions.cpp): a CUDA conversion works on device
// buffers, and the Vulkan artifact carries no host loops.
void convertThreshold(int srcCode, void *x, sd::LongType N, int dstCode, void *z) {
#if defined(SD_CUDA) || defined(SD_VULKAN)
  rejectConversion(srcCode, dstCode, "this backend has no threshold encoding");
#else
  const bool encode = dstCode == CONVERSION_THRESHOLD;
  switch (conversionElementType(encode ? srcCode : dstCode)) {
    case sd::DataType::HALF:
      if (encode)
        sd::TypeCast::convertToThreshold<float16>(nullptr, x, N, z);
      else
        sd::TypeCast::convertFromThreshold<float16>(nullptr, x, N, z);
      return;
    case sd::DataType::FLOAT32:
      if (encode)
        sd::TypeCast::convertToThreshold<float>(nullptr, x, N, z);
      else
        sd::TypeCast::convertFromThreshold<float>(nullptr, x, N, z);
      return;
    case sd::DataType::DOUBLE:
      if (encode)
        sd::TypeCast::convertToThreshold<double>(nullptr, x, N, z);
      else
        sd::TypeCast::convertFromThreshold<double>(nullptr, x, N, z);
      return;
    default:
      rejectConversion(srcCode, dstCode, "the threshold encoding takes FLOAT16, FLOAT or DOUBLE arrays");
      return;
  }
#endif
}

}  // namespace

/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, int srcType, Pointer x, long N, int dstType, Pointer z);
 *
 * srcType and dstType are DataTypeEx ordinals (ConversionCode above). On CUDA x and z are device buffers converted on
 * the stream in extras[1]; on the other backends they are host buffers.
 */
void convertTypes(sd::Pointer *extras, int srcType, sd::Pointer hX, sd::LongType N, int dstType, sd::Pointer hZ) {
  try {
    if (srcType == CONVERSION_THRESHOLD || dstType == CONVERSION_THRESHOLD) {
      if (srcType == dstType) {
        rejectConversion(srcType, dstType, "nothing to encode or decode");
        return;
      }
      convertThreshold(srcType, hX, N, dstType, hZ);
      return;
    }

    const sd::DataType srcElement = conversionElementType(srcType);
    const sd::DataType dstElement = conversionElementType(dstType);
    if (srcElement == sd::DataType::UNKNOWN || dstElement == sd::DataType::UNKNOWN) {
      rejectConversion(srcType, dstType, "not an element type conversion");
      return;
    }
    if (N <= 0 || (srcElement == dstElement && hX == hZ)) return;

#if defined(SD_CUDA)
    if (extras == nullptr || extras[1] == nullptr) {
      rejectConversion(srcType, dstType, "a CUDA conversion takes its stream in extras[1]");
      return;
    }
    BUILD_DOUBLE_SELECTOR(srcElement, dstElement, sd::TypeCast::convertGenericCuda, (extras, hX, N, hZ),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
    BUILD_DOUBLE_SELECTOR(srcElement, dstElement, sd::TypeCast::convertGeneric, (nullptr, hX, N, hZ), SD_COMMON_TYPES,
                          SD_COMMON_TYPES);
#endif
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
