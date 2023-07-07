//
// Created by agibsonccc on 7/7/23.
//
#include "DeviceValidator.h"


  DeviceValidator::DeviceValidator(int device) {
    cudaGetDeviceProperties(&prop, device);
  }

  template <typename... Args>
  KernelFuncPtr<Args...> getKernelFuncPtr(void (*kernelFunc)(Args...)) {
    return reinterpret_cast<KernelFuncPtr<Args...>>(kernelFunc);
  }



  ValidationResult DeviceValidator::validateKernelLaunch(const char* name, void* funcHandle, dim3 threadsPerBlock, dim3 numBlocks, size_t globalMemoryUsage, int minComputeCapability) {
    ValidationResult result;
    CUfunction kernel;
    if (funcHandle != nullptr) {
      kernel = reinterpret_cast<CUfunction>(funcHandle);
    } else {
      cuModuleGetFunction(&kernel, nullptr, name);
    }
    int sharedSizeBytes, numRegs, maxThreadsPerBlock;
    cuFuncGetAttribute(&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
    cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);

    result.isComputeCapabilitySufficient = prop.major * 10 + prop.minor >= minComputeCapability;
    result.isECCMemorySupported = prop.ECCEnabled;
    result.isManagedMemorySupported = prop.managedMemory;
    result.isComputePreemptionSupported = prop.computePreemptionSupported;
    result.isConcurrentKernelsSupported = prop.concurrentKernels;
    result.isL2CacheSizeSufficient = prop.l2CacheSize;
    result.isThreadsPerBlockWithinLimit = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= prop.maxThreadsPerBlock;
    result.isThreadsPerBlockWithinLimit &= threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= maxThreadsPerBlock;
    result.isBlocksWithinGridSizeLimit = numBlocks.x <= prop.maxGridSize[0] && numBlocks.y <= prop.maxGridSize[1] && numBlocks.z <= prop.maxGridSize[2];
    result.isSharedMemoryUsageWithinLimit = sharedSizeBytes <= prop.sharedMemPerBlock;
    result.isRegisterUsageWithinLimit = numRegs * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= prop.regsPerBlock;
    result.isTotalThreadsWithinLimit = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * numBlocks.x * numBlocks.y * numBlocks.z <= prop.maxThreadsPerMultiProcessor;
    result.isGlobalMemoryUsageWithinLimit = globalMemoryUsage <= prop.totalGlobalMem;

    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    size_t usedMemory = totalMemory - freeMemory;
    result.isMemoryUsageWithinLimit = usedMemory + globalMemoryUsage <= totalMemory;
          result.isLocalMemoryUsageWithinLimit = usedMemory + globalMemoryUsage <= prop.localL1CacheSupported ? totalMemory : prop.localL1CacheSupported;
    return result;
  }




  void DeviceValidator::setKernelAttribute(const char* name, void* funcHandle, CUfunction_attribute attribute, int value) {
    CUfunction kernel;
    if (funcHandle != nullptr) {
      kernel = reinterpret_cast<CUfunction>(funcHandle);
    } else {
      cuModuleGetFunction(&kernel, nullptr, name);
    }
    cuFuncSetAttribute(kernel, attribute, value);
  }

  void DeviceValidator::setKernelMaxDynamicSharedSizeBytes(const char* name,void* funcHandle, int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value);
  }

  void DeviceValidator::setKernelPreferredSharedMemoryCarveout(const char* name, void *funcHandle,int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value);
  }

  void DeviceValidator::setKernelMaxRegisters(const char* name,void *funcHandle, int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_MAX, value);
  }

  void DeviceValidator::setKernelMaxThreadsPerBlock(const char* name, void *funcHandle,int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, value);
  }

  void DeviceValidator::setKernelNumRegs(const char* name, void *funcHandle,int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_NUM_REGS, value);
  }

  void DeviceValidator::setKernelSharedSizeBytes(const char* name, void *funcHandle,int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
  }

  void DeviceValidator::setKernelBinaryVersion(const char* name, void *funcHandle,int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_BINARY_VERSION, value);
  }

  void DeviceValidator::setKernelCacheModeCA(const char* name,void *funcHandle, int value) {
    setKernelAttribute(name,funcHandle, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, value);
  }



  void DeviceValidator::setKernelMaxThreadsPerBlockOptIn(const char* name,void *funcHandle, int value) {
    setKernelAttribute(name, funcHandle,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,value);
  }

  void DeviceValidator::setKernelReservedSharedSizeBytes(const char* name,void *funcHandle, int value) {
    setKernelAttribute(name, funcHandle,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
  }
