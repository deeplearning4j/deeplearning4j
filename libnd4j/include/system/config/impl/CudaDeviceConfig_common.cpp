/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <system/config/CudaDeviceConfig.h>

namespace sd {
namespace config {

// Environment owns every subsystem configuration object on every backend.
// Construction only applies the header-defined defaults; CUDA runtime queries
// remain in CudaDeviceConfig.cpp and are invoked only by an SD_CUDA build.
CudaDeviceConfig::CudaDeviceConfig() = default;

}  // namespace config
}  // namespace sd
