/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_EXTRAARGUMENTS_DEVICE_H
#define LIBND4J_EXTRAARGUMENTS_DEVICE_H

#include <system/common.h>
#include <system/BackendNamespace.h>

#include <cstddef>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace extra_args_detail {

/** Allocate bytes in the selected backend's current-device address space. */
SD_LIB_EXPORT void* extraArgsAllocDevice(size_t bytes);

/** Release a backend allocation with the active execution stream's ordering. */
SD_LIB_EXPORT void extraArgsFreeDevice(void* ptr);

/** Copy host bytes to a backend allocation with active-stream ordering. */
SD_LIB_EXPORT void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes);

}  // namespace extra_args_detail
SD_BACKEND_ABI_NAMESPACE_END

#endif  // LIBND4J_EXTRAARGUMENTS_DEVICE_H
