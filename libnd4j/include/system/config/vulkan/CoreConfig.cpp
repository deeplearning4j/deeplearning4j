/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "system/config/vulkan/CoreConfig.cpp is only valid for SD_VULKAN"
#endif

#include <system/config/CoreConfig.h>

namespace sd {
namespace config {

bool CoreConfig::isSerializeBlasCalls() {
  return _serializeBlasCalls.load();
}

void CoreConfig::setSerializeBlasCalls(bool serialize) {
  _serializeBlasCalls.store(serialize);
  _serializeBlasCallsSet.store(true);
}

int CoreConfig::getOpenBlasThreads() {
  return _openBlasThreads.load();
}

void CoreConfig::setOpenBlasThreads(int threads) {
  _openBlasThreads.store(threads);
}

}  // namespace config
}  // namespace sd
