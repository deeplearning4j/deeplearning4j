/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/ExtraArguments_device.h>

#include <cstdint>
#include <cstring>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace extra_args_detail {

void* extraArgsAllocDevice(size_t bytes) {
  return new int8_t[bytes];
}

void extraArgsFreeDevice(void* ptr) {
  delete[] reinterpret_cast<int8_t*>(ptr);
}

void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

}  // namespace extra_args_detail
SD_BACKEND_ABI_NAMESPACE_END
