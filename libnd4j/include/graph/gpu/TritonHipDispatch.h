/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_TRITON_HIP_DISPATCH_H
#define LIBND4J_TRITON_HIP_DISPATCH_H

// Match the existing Triton AMD runtime admission without exposing HIP headers
// to the CUDA-facing dispatcher.
#if defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN) || defined(SD_HIP)
#define TRITON_HAS_HIP 1

#include <string>

namespace sd {
namespace graph {
namespace triton_hip {

// SDK-neutral boundary: HIP owns all native types and casts in the implementation.
// Handles, stream ordering and kernel argument storage are passed through unchanged.
// On failure, error receives HIP's runtime-owned error string (no caller ownership).
bool detectDevice(std::string& archName, std::string& deviceName);
void* loadModule(const void* data, const char*& error);
void* getKernelFunction(void* module, const char* name, const char*& error);
bool launchKernel(void* function,
                  unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                  unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                  unsigned int sharedMemBytes, void* stream, void** args,
                  const char*& error);
void unloadModule(void* module);

}  // namespace triton_hip
}  // namespace graph
}  // namespace sd

#endif  // HIP runtime available
#endif  // LIBND4J_TRITON_HIP_DISPATCH_H
