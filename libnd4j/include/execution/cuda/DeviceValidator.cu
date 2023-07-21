#include "DeviceValidator.h"

ValidationResult::ValidationResult()
    : isComputeCapabilitySufficient(true),
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



DeviceValidator* DeviceValidator::instance = nullptr;
std::mutex DeviceValidator::mtx;
DeviceValidator* DeviceValidator::getInstance(const std::string& directory, int device) {
  std::lock_guard<std::mutex> lock(mtx);
  if (instance == nullptr) {
    instance = new DeviceValidator(directory, device);
  }
  return instance;
}

DeviceValidator::DeviceValidator(const std::string& directoryPath, int device)
    : directoryPath(directoryPath) {
  cudaGetDeviceProperties(&prop, device);
  init();
}

DeviceValidator::~DeviceValidator() {
  for (auto& pair : moduleMap) {
    cuModuleUnload(pair.second);
  }
  moduleMap.clear();
  functionMap.clear();
}



std::vector<std::string> DeviceValidator::parsePTXFile(const std::string& filePath) {
  std::ifstream file(filePath);
  std::string line;
  std::vector<std::string> functionNames;
  std::regex functionPattern("\\.entry\\s+([a-zA-Z_][a-zA-Z0-9_]*)");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, functionPattern) && match.size() > 1) {
      functionNames.push_back(match.str(1));
    }
  }

  return functionNames;
}

std::vector<std::string> DeviceValidator::parseCUBINFile(const std::string& filePath) {
  std::string command = "cuobjdump -sass " + filePath + " | grep -oP '(?<=FUNC ).*(?=\\()'";
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  std::stringstream ss(result);
  std::string functionName;
  std::vector<std::string> functionNames;
  while (std::getline(ss, functionName, '\n')) {
    functionNames.push_back(functionName);
  }
  return functionNames;
}

void DeviceValidator::init() {
// Check if cuobjdump is available
#ifdef _WIN32
  // Windows
  if (system("cuobjdump --version > nul 2>&1") != 0) {
    throw std::runtime_error("cuobjdump is not available on the system's PATH. Please install it and try again.");
  }
#else
  // Linux
  if (system("cuobjdump --version > /dev/null 2>&1") != 0) {
    throw std::runtime_error("cuobjdump is not available on the system's PATH. Please install it and try again.");
  }
#endif

  printf("Initializing DeviceValidator at path %s\n", directoryPath.c_str());
  for (const auto &entry : std::filesystem::directory_iterator(directoryPath)) {
    if (entry.path().extension() == ".ptx" || entry.path().extension() == ".cubin") {
      CUmodule cuModule;
      printf("Adding path %s\n",entry.path().filename().c_str());
      if (cuModuleLoad(&cuModule, entry.path().c_str()) == CUDA_SUCCESS) {
        moduleMap[entry.path().filename().string()] = cuModule;

        if (entry.path().extension() == ".ptx") {
          std::vector<std::string> functionNames = parsePTXFile(entry.path().c_str());
          for (const auto& functionName : functionNames) {
            CUfunction cuFunction;
            if (cuModuleGetFunction(&cuFunction, cuModule, functionName.c_str()) == CUDA_SUCCESS) {
              functionMap[functionName] = cuFunction;
            }
          }
        } else if (entry.path().extension() == ".cubin") {
          std::vector<std::string> functionNames = parseCUBINFile(entry.path().c_str());
          for (const auto& functionName : functionNames) {
            CUfunction cuFunction;
            if (cuModuleGetFunction(&cuFunction, cuModule, functionName.c_str()) == CUDA_SUCCESS) {
              functionMap[functionName] = cuFunction;
            }
          }
        }
      }
    }
  }
}

void DeviceValidator::setKernelAttribute(const std::string& functionName, CUfunction_attribute attribute, int value) {
  if (functionMap.find(functionName) != functionMap.end()) {
    cuFuncSetAttribute(functionMap[functionName], attribute, value);
  }
}

void DeviceValidator::setKernelMaxDynamicSharedSizeBytes(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value);
}

void DeviceValidator::setKernelPreferredSharedMemoryCarveout(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value);
}

void DeviceValidator::setKernelMaxRegisters(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_NUM_REGS, value);
}

void DeviceValidator::setKernelMaxThreadsPerBlock(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, value);
}

void DeviceValidator::setKernelNumRegs(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_NUM_REGS, value);
}

void DeviceValidator::setKernelSharedSizeBytes(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
}

void DeviceValidator::setKernelBinaryVersion(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_BINARY_VERSION, value);
}

void DeviceValidator::setKernelCacheModeCA(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, value);
}

void DeviceValidator::setKernelMaxThreadsPerBlockOptIn(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, value);
}

void DeviceValidator::setKernelReservedSharedSizeBytes(const std::string& functionName, int value) {
  setKernelAttribute(functionName, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
}


void DeviceValidator::setAllKernelsAttribute(CUfunction_attribute attribute, int value) {
  for (auto& pair : functionMap) {
    cuFuncSetAttribute(pair.second, attribute, value);
  }
}

void DeviceValidator::setAllKernelsMaxDynamicSharedSizeBytes(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value);
}

void DeviceValidator::setAllKernelsPreferredSharedMemoryCarveout(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value);
}

void DeviceValidator::setAllKernelsMaxRegisters(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, value);
}

void DeviceValidator::setAllKernelsMaxThreadsPerBlock(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, value);
}

void DeviceValidator::setAllKernelsNumRegs(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, value);
}

void DeviceValidator::setAllKernelsSharedSizeBytes(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
}

void DeviceValidator::setAllKernelsBinaryVersion(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION, value);
}

void DeviceValidator::setAllKernelsCacheModeCA(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, value);
}

void DeviceValidator::setAllKernelsMaxThreadsPerBlockOptIn(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, value);
}

void DeviceValidator::setAllKernelsReservedSharedSizeBytes(int value) {
  setAllKernelsAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, value);
}

void DeviceValidator::printKernelAttribute(const char* name, CUfunction_attribute attribute) {
  CUfunction kernel = functionMap[name];
  int value;
  cuFuncGetAttribute(&value, attribute, kernel);
  std::cout << "Attribute " << attribute << " for function " << name << " is " << value << std::endl;
}

ValidationResult DeviceValidator::validateKernelLaunch(const char* name, dim3 threadsPerBlock, dim3 numBlocks,
                                                       size_t globalMemoryUsage) {
  ValidationResult result;
  CUfunction kernel;

  // Look up the function from the function map
  auto it = functionMap.find(name);
  if (it != functionMap.end()) {
    kernel = it->second;
  } else {
    // If the function is not found in the map, return an invalid ValidationResult
    return result;
  }

  int sharedSizeBytes, numRegs, maxThreadsPerBlock;
  cuFuncGetAttribute(&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
  cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);

  result.sharedSizeBytes = sharedSizeBytes;
  result.numRegs = numRegs;
  result.maxThreadsPerBlock = maxThreadsPerBlock;

  result.isComputeCapabilitySufficient = true;
  result.isECCMemorySupported = prop.ECCEnabled;
  result.isManagedMemorySupported = prop.managedMemory;
  result.isComputePreemptionSupported = prop.computePreemptionSupported;
  result.isConcurrentKernelsSupported = prop.concurrentKernels;
  result.isL2CacheSizeSufficient = prop.l2CacheSize;
  result.isThreadsPerBlockWithinLimit = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= prop.maxThreadsPerBlock;
  result.isThreadsPerBlockWithinLimit &= threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= maxThreadsPerBlock;
  result.numBlocks = numBlocks.x;
  result.isBlocksWithinGridSizeLimit = numBlocks.x <= prop.maxGridSize[0] && numBlocks.y <= prop.maxGridSize[1] && numBlocks.z <= prop.maxGridSize[2];
  result.isSharedMemoryUsageWithinLimit = sharedSizeBytes <= prop.sharedMemPerBlock;
  result.isRegisterUsageWithinLimit = numRegs * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= prop.regsPerBlock;
  result.numThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * numBlocks.x * numBlocks.y * numBlocks.z;
  result.isTotalThreadsWithinLimit = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * numBlocks.x * numBlocks.y * numBlocks.z <= prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
  result.isGlobalMemoryUsageWithinLimit = globalMemoryUsage <= prop.totalGlobalMem;
  result.globalMemory = globalMemoryUsage;
  size_t freeMemory, totalMemory;
  cudaMemGetInfo(&freeMemory, &totalMemory);
  size_t usedMemory = totalMemory - freeMemory;
  result.memoryUsage = usedMemory + globalMemoryUsage;
  result.isMemoryUsageWithinLimit = result.memoryUsage <= totalMemory;
  result.isLocalMemoryUsageWithinLimit = usedMemory + globalMemoryUsage <= prop.localL1CacheSupported ? totalMemory : prop.localL1CacheSupported;
  result.freeMemory = freeMemory;
  result.totalMemory = totalMemory;

  return result;
}

void DeviceValidator::printProblematicFunctions(dim3 threadsPerBlock, dim3 numBlocks, size_t globalMemory) {
  printf("Attempting to print functions with size of map %d\n",functionMap.size());
  for (const auto& pair : functionMap) {
    const std::string functionName = pair.first;
    CUfunction function = pair.second;

    ValidationResult validationResult = validateKernelLaunch(functionName.c_str(),threadsPerBlock, numBlocks, globalMemory);

    if (!validationResult.isValid()) {
      std::cout << "Function " << functionName << " has the following problems:\n";
      printValidationResult(functionName.c_str(),validationResult);
      std::cout << std::endl;
    }
  }
}

bool ValidationResult::isValid()  {
  return isComputeCapabilitySufficient &&
         isManagedMemorySupported &&
         isComputePreemptionSupported &&
         isThreadsPerBlockWithinLimit &&
         isBlocksWithinGridSizeLimit &&
         isSharedMemoryUsageWithinLimit &&
         isRegisterUsageWithinLimit &&
         isTotalThreadsWithinLimit &&
         isGlobalMemoryUsageWithinLimit &&
         isMemoryUsageWithinLimit &&
         isLocalMemoryUsageWithinLimit &&
         isConcurrentKernelsSupported &&
         isL2CacheSizeSufficient;
}

void DeviceValidator::printValidationResult(const char* name, ValidationResult& result) {
  printf("Validating: %s\n",name);
  if (!result.isValid()) {
    std::cout << "Function " << name << " has an issue:\n";
    if (!result.isComputeCapabilitySufficient) {
      std::cout << " - Compute capability is not sufficient. Required: " << result.isComputeCapabilitySufficient << ", Actual: " << prop.major * 10 + prop.minor << "\n";
    }

    if (!result.isManagedMemorySupported) {
      std::cout << " - Managed memory is not supported.\n";
    }
    if (!result.isComputePreemptionSupported) {
      std::cout << " - Compute preemption is not supported.\n";
    }
    if (!result.isThreadsPerBlockWithinLimit) {
      std::cout << " - Threads per block is not within limit. Max: " << prop.maxThreadsPerBlock << "\n";
      std::cout << " - Value is: " <<  result.maxThreadsPerBlock << "\n";

    }
    if (!result.isBlocksWithinGridSizeLimit) {
      //  result.isRegisterUsageWithinLimit = numRegs * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= prop.regsPerBlock;
      std::cout << " - Blocks within grid size is not within limit. Max: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
      std::cout << " - Value is: (" << result.numBlocks << ", " << result.numBlocks << ", " << result.numBlocks << ")\n";
    }
    if (!result.isSharedMemoryUsageWithinLimit) {
      std::cout << " - Shared memory usage is not within limit. Max: " << prop.sharedMemPerBlock << "\n";
    }
    if (!result.isRegisterUsageWithinLimit) {
      std::cout << " - Register usage is not within limit. Max: " << prop.regsPerBlock << "\n";
      std::cout << " - Value is: (" << result.numRegs << ", " << result.numBlocks << "\n";
    }
    if (!result.isTotalThreadsWithinLimit) {
      std::cout << " - Total threads is not within limit. Max: " << prop.maxThreadsPerMultiProcessor *  prop.multiProcessorCount << "\n";
      std::cout << " - Value is: " << result.numThreads << "\n";
    }
    if (!result.isGlobalMemoryUsageWithinLimit) {
      std::cout << " - Global memory usage is not within limit. Max: " << prop.totalGlobalMem << "\n";
      std::cout << " - Value is: " << result.globalMemory << "\n";
    }
    if (!result.isMemoryUsageWithinLimit) {
      std::cout << " - Memory usage is not within limit.\n";
      std::cout << " - Value is: " << result.memoryUsage << "\n";
    }
    if (!result.isLocalMemoryUsageWithinLimit) {
      std::cout << " - Local memory usage is not within limit. Max: " << (prop.localL1CacheSupported ? prop.localL1CacheSupported : prop.totalGlobalMem) << "\n";
    }
    if (!result.isConcurrentKernelsSupported) {
      std::cout << " - Concurrent kernels is not supported.\n";
    }
    if (!result.isL2CacheSizeSufficient) {
      std::cout << " - L2 cache size is not sufficient. Max: " << prop.l2CacheSize << "\n";
    }
  }
}

void DeviceValidator::printMaxKernelAttributes() {
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
  std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << std::endl;
  std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
  std::cout << "Warp size: " << prop.warpSize << std::endl;
  std::cout << "Max pitch: " << prop.memPitch << std::endl;
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
  std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
  std::cout << "Clock rate: " << prop.clockRate << std::endl;
  std::cout << "Total constant memory: " << prop.totalConstMem << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Texture alignment: " << prop.textureAlignment << std::endl;
  std::cout << "Concurrent copy and execution: " << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
  std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Kernel execution timeout: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
  std::cout << "Integrated: " << (prop.integrated ? "Yes" : "No") << std::endl;
  std::cout << "Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
  std::cout << "Compute mode: " << (prop.computeMode == cudaComputeModeDefault ? "Default" : (prop.computeMode == cudaComputeModeExclusive ? "Exclusive" : (prop.computeMode == cudaComputeModeProhibited ? "Prohibited" : "Exclusive Process"))) << std::endl;
  std::cout << "Max texture 1D: " << prop.maxTexture1D << std::endl;
  std::cout << "Max texture 2D: (" << prop.maxTexture2D[0] << ", " << prop.maxTexture2D[1] << ")" << std::endl;
  std::cout << "Max texture 3D: (" << prop.maxTexture3D[0] << ", " << prop.maxTexture3D[1] << ", " << prop.maxTexture3D[2] << ")" << std::endl;
  std::cout << "Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
  std::cout << "ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
  std::cout << "PCI bus ID: " << prop.pciBusID << std::endl;
  std::cout << "PCI device ID: " << prop.pciDeviceID << std::endl;
  std::cout << "TCC driver: " << (prop.tccDriver ? "Yes" : "No") << std::endl;
  std::cout << "Memory clock rate: " << prop.memoryClockRate << std::endl;
  std::cout << "Global memory bus width: " << prop.memoryBusWidth << std::endl;
  std::cout << "L2 cache size: " << prop.l2CacheSize << std::endl;
  std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Stream priorities supported: " << (prop.streamPrioritiesSupported ? "Yes" : "No") << std::endl;
  std::cout << "Global L1 cache supported: " << (prop.globalL1CacheSupported ? "Yes" : "No") << std::endl;
  std::cout << "Local L1 cache supported: " << (prop.localL1CacheSupported ? "Yes" : "No") << std::endl;
  std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
  std::cout << "Registers per multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
  std::cout << "Managed memory concurrent: " << (prop.concurrentManagedAccess ? "Yes" : "No") << std::endl;
  std::cout << "Is multi GPU board: " << (prop.isMultiGpuBoard ? "Yes" : "No") << std::endl;
  std::cout << "Multi GPU board group ID: " << prop.multiGpuBoardGroupID << std::endl;
  std::cout << "Host native atomic supported: " << (prop.hostNativeAtomicSupported ? "Yes" : "No") << std::endl;
  std::cout << "Single to double precision perf ratio: " << prop.singleToDoublePrecisionPerfRatio << std::endl;
  std::cout << "Pageable memory access: " << (prop.pageableMemoryAccess ? "Yes" : "No") << std::endl;
  std::cout << "Concurrent managed access: " << (prop.concurrentManagedAccess ? "Yes" : "No") << std::endl;
}


void DeviceValidator::printKernelAttributes(const char* name) {
  auto it = functionMap.find(name);
  if (it == functionMap.end()) {
    std::cerr << "Error: Function " << name << " not found." << std::endl;
    return;
  }
  CUfunction kernel = it->second;

  int sharedSizeBytes, numRegs, maxThreadsPerBlock, binaryVersion, cacheModeCA, maxDynamicSharedSizeBytes, preferredSharedMemoryCarveout;
  cuFuncGetAttribute(&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
  cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
  cuFuncGetAttribute(&binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kernel);
  cuFuncGetAttribute(&cacheModeCA, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel);
  cuFuncGetAttribute(&maxDynamicSharedSizeBytes, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel);
  cuFuncGetAttribute(&preferredSharedMemoryCarveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kernel);

  std::cout << "Kernel name: " << name << std::endl;
  std::cout << "Shared memory per block: " << sharedSizeBytes << std::endl;
  std::cout << "Registers per block: " << numRegs << std::endl;
  std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
  std::cout << "Binary version: " << binaryVersion << std::endl;
  std::cout << "Cache mode CA: " << cacheModeCA << std::endl;
  std::cout << "Max dynamic shared size bytes: " << maxDynamicSharedSizeBytes << std::endl;
  std::cout << "Preferred shared memory carveout: " << preferredSharedMemoryCarveout << std::endl;

  if (sharedSizeBytes > prop.sharedMemPerBlock) {
    std::cout << "WARNING: The kernel uses more shared memory per block than the device supports." << std::endl;
  }
  if (numRegs > prop.regsPerBlock) {
    std::cout << "WARNING: The kernel uses more registers per block than the device supports." << std::endl;
  }
  if (maxThreadsPerBlock > prop.maxThreadsPerBlock) {
    std::cout << "WARNING: The kernel uses more threads per block than the device supports." << std::endl;
  }
  // Note: There are no device properties to compare with binaryVersion, cacheModeCA, maxDynamicSharedSizeBytes, and preferredSharedMemoryCarveout.
}


std::map<std::string, ValidationResult> DeviceValidator::collectResourceProblems() {
  std::map<std::string, ValidationResult> problematicFunctions;
  for (const auto& pair : functionMap) {
    const char* name = pair.first.c_str();
    ValidationResult result = validateKernelLaunch(name, dim3(1, 1, 1), dim3(1, 1, 1), 0);
    if (!result.isComputeCapabilitySufficient ||
        !result.isECCMemorySupported ||
        !result.isManagedMemorySupported ||
        !result.isComputePreemptionSupported ||
        !result.isThreadsPerBlockWithinLimit ||
        !result.isBlocksWithinGridSizeLimit ||
        !result.isSharedMemoryUsageWithinLimit ||
        !result.isRegisterUsageWithinLimit ||
        !result.isTotalThreadsWithinLimit ||
        !result.isGlobalMemoryUsageWithinLimit ||
        !result.isMemoryUsageWithinLimit ||
        !result.isLocalMemoryUsageWithinLimit ||
        !result.isConcurrentKernelsSupported ||
        !result.isL2CacheSizeSufficient) {
      problematicFunctions[name] = result;
    }
  }
  return problematicFunctions;
}