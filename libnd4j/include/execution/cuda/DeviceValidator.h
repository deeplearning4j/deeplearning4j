//
// Created by agibsonccc on 7/7/23.
//

#ifndef LIBND4J_DEVICEVALIDATOR_H
#define LIBND4J_DEVICEVALIDATOR_H
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>
#include <cuda.h>

class ValidationResult {
 public:
  bool isComputeCapabilitySufficient;
  bool isECCMemorySupported;
  bool isManagedMemorySupported;
  bool isComputePreemptionSupported;
  bool isThreadsPerBlockWithinLimit;
  bool isBlocksWithinGridSizeLimit;
  bool isSharedMemoryUsageWithinLimit;
  bool isRegisterUsageWithinLimit;
  bool isTotalThreadsWithinLimit;
  bool isGlobalMemoryUsageWithinLimit;
  bool isMemoryUsageWithinLimit;
  bool isLocalMemoryUsageWithinLimit;
  bool isConcurrentKernelsSupported;
  bool isL2CacheSizeSufficient;

  ValidationResult()
      : isComputeCapabilitySufficient(true),
        isECCMemorySupported(true),
        isManagedMemorySupported(true),
        isComputePreemptionSupported(true),
        isThreadsPerBlockWithinLimit(true),
        isBlocksWithinGridSizeLimit(true),
        isSharedMemoryUsageWithinLimit(true),
        isRegisterUsageWithinLimit(true),
        isTotalThreadsWithinLimit(true),
        isGlobalMemoryUsageWithinLimit(true),
        isMemoryUsageWithinLimit(true),
        isLocalMemoryUsageWithinLimit(true),
        isConcurrentKernelsSupported(true),
        isL2CacheSizeSufficient(true) {}
};

// Define a function pointer type for your kernel
template <typename... Args>
using KernelFuncPtr = void (*)(Args...);


class DeviceValidator {
 private:
  cudaDeviceProp prop;

 public:


  // Function to get a void* from a kernel function
  template <typename... Args>
  using KernelFuncPtr = void (*)(Args...);

  template <typename... Args>
  KernelFuncPtr<Args...> getKernelFuncPtr(void (*kernelFunc)(Args...));


  DeviceValidator(int device = 0);

  ValidationResult validateKernelLaunch(const char* name, void* funcHandle,
                                        dim3 threadsPerBlock, dim3 numBlocks,
                                        size_t globalMemoryUsage,
                                        int minComputeCapability);

  void setKernelAttribute(const char* name, void* funcHandle, CUfunction_attribute attribute, int value);


  void setKernelMaxDynamicSharedSizeBytes(const char* name, void* funcHandle, int value);

  void setKernelPreferredSharedMemoryCarveout(const char* name, void* funcHandle, int value);

  void setKernelMaxRegisters(const char* name, void* funcHandle, int value);

  void setKernelMaxThreadsPerBlock(const char* name, void* funcHandle, int value);

  void setKernelNumRegs(const char* name, void* funcHandle, int value);

  void setKernelSharedSizeBytes(const char* name, void* funcHandle, int value);

  void setKernelBinaryVersion(const char* name, void* funcHandle, int value);

  void setKernelCacheModeCA(const char* name, void* funcHandle, int value);

  void setKernelMaxThreadsPerBlockOptIn(const char* name, void* funcHandle, int value);

  void setKernelReservedSharedSizeBytes(const char* name, void* funcHandle, int value);
};

#endif  // LIBND4J_DEVICEVALIDATOR_H
