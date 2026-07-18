/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <helpers/DebugHelper.h>
#include <legacy/NativeOps.h>

#if defined(SD_VULKAN)
#error "Vulkan inspectArray must be provided by the Vulkan device ABI"
#endif


void inspectArray(sd::Pointer *extraPointers, sd::Pointer buffer, sd::LongType *shapeInfo,
                  sd::Pointer specialBuffer, sd::LongType *specialShapeInfo, sd::Pointer debugInfo) {
#ifdef __cpp_exceptions
  try {
    auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
    sd::NDArray array(buffer, shapeInfo, nullptr, 0, 0);
    sd::DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    THROW_EXCEPTION(e.what());
  }
#else
  auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
  sd::NDArray array(buffer, shapeInfo, nullptr, 0, 0);
  sd::DebugHelper::retrieveDebugStatistics(p, &array);
#endif
}

