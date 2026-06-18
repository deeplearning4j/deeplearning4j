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

//
// Split from NativeOpsHelpers_DataBuffers.cpp to reduce object file size
// Contains: TAD pack functions
//

#include <legacy/NativeOps.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/shape.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>

// TadPack lifetime registry - keeps shared_ptr<TadPack> alive for TadPacks returned to Java
// Without this, when ConstantTadHelper::tadForDimensions() returns shared_ptr<TadPack>,
// but tadOnlyShapeInfo() returns raw TadPack*, the local shared_ptr goes out of scope
// and TadPack can be deleted while Java still holds the raw pointer -> SIGSEGV
std::unordered_map<sd::TadPack*, std::shared_ptr<sd::TadPack>> g_tadPackRegistry;
std::mutex g_tadPackMutex;


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

const char* getTadPackStackTrace(OpaqueTadPack *pack) {
  if (pack == nullptr) {
    return "TadPack is null";
  }

  //
  // ROOT CAUSE: thread_local uses R_X86_64_GOTPC32_TLSDESC relocations which have +/-2GB limit
  // When SD_GCC_FUNCTRACE is enabled, binary size exceeds 2GB -> TLS relocations fail
  //
  // SOLUTION: Use regular static instead of thread_local
  // - Eliminates all TLS relocations from this function
  // - Trade-off: Not thread-safe (acceptable for debugging function)
  // - If called concurrently by multiple threads, traces may interleave (rare edge case)
  //
  static std::string cachedTrace;
  cachedTrace = pack->getStackTraceAsString();

  return cachedTrace.c_str();
}


sd::TadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength) {
#ifdef __cpp_exceptions
  try {
    if(hXShapeInfo->primary() == nullptr) {
      THROW_EXCEPTION("tadOnlyShapeInfo: hXShapeInfo->primary() is nullptr!");
    }

    auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());

    sd::LongType firstValue = buffPrim[0];
    if (firstValue < 0 || firstValue > SD_MAX_RANK) {
      std::string errorMessage = "tadOnlyShapeInfo: Shape buffer contains invalid rank value: ";
      errorMessage += std::to_string(firstValue);
      errorMessage += " (0x";
      char hexBuf[32];
      snprintf(hexBuf, sizeof(hexBuf), "%lx", static_cast<unsigned long>(firstValue));
      errorMessage += hexBuf;
      errorMessage += "). ";

      if (firstValue > 0x10000000000ULL) {
        errorMessage += "This value looks like a memory address, suggesting corruption. ";
      }

      THROW_EXCEPTION(errorMessage.c_str());
    }

    auto shapeFromCache = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(buffPrim)->primary();
    auto rankVal = shapeFromCache[0];
    if(rankVal == 0) {
      auto len = shape::shapeInfoLength(rankVal);
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

    if (!pack) {
      THROW_EXCEPTION("tadOnlyShapeInfo: Failed to create TadPack!");
    }

    sd::TadPack* rawPtr = pack.get();

    {
      std::lock_guard<std::mutex> lock(g_tadPackMutex);
      g_tadPackRegistry[rawPtr] = pack;
    }

    return rawPtr;
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    THROW_EXCEPTION(e.what());
  }
#else
  if(hXShapeInfo->primary() == nullptr) {
    safeSetErrorContext(1, "tadOnlyShapeInfo: hXShapeInfo->primary() is nullptr!");
    return nullptr;
  }

  auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());

  sd::LongType firstValue = buffPrim[0];
  if (firstValue < 0 || firstValue > SD_MAX_RANK) {
    std::string errorMessage = "tadOnlyShapeInfo: Shape buffer contains invalid rank value: ";
    errorMessage += std::to_string(firstValue);
    safeSetErrorContext(1, errorMessage.c_str());
    return nullptr;
  }

  auto shapeFromCache = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(buffPrim)->primary();
  auto rankVal = shapeFromCache[0];
  if(rankVal == 0) {
    auto len = shape::shapeInfoLength(rankVal);
    bool allZero = true;
    for(int i = 0; i < len; i++) {
      if(buffPrim[i] != 0) {
        allZero = false;
        break;
      }
    }

    if(allZero) {
      safeSetErrorContext(1, "Found shape buffer with all zero values. Values likely unset.");
      return nullptr;
    }
  }

  auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(
      shapeFromCache, dimension, dimensionLength);

  if (!pack) {
    safeSetErrorContext(1, "tadOnlyShapeInfo: Failed to create TadPack!");
    return nullptr;
  }

  sd::TadPack* rawPtr = pack.get();

  {
    std::lock_guard<std::mutex> lock(g_tadPackMutex);
    g_tadPackRegistry[rawPtr] = pack;
  }

  return rawPtr;
#endif

  return nullptr;
}

// Helper function to clear the TadPack registry
void clearTadPackRegistry() {
  std::lock_guard<std::mutex> lock(g_tadPackMutex);
  g_tadPackRegistry.clear();
}

