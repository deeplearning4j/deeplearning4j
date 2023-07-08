#include <cuda.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

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
  bool isValid();

  ValidationResult();
};

class DeviceValidator {
 private:
  cudaDeviceProp prop;
  static std::mutex mtx;
  std::unordered_map<std::string, CUmodule> moduleMap;
  std::unordered_map<std::string, CUfunction> functionMap;
  std::string directoryPath;
  static DeviceValidator* instance;

  void init();

 public:
  DeviceValidator(const std::string& directoryPath, int device = 0);
  ~DeviceValidator();
  static DeviceValidator* getInstance(const std::string& directory, int device = 0);

  // Set kernel attribute
  void setKernelAttribute(const std::string& functionName, CUfunction_attribute attribute, int value);

  void setKernelMaxDynamicSharedSizeBytes(const std::string& functionName, int value);

  void setKernelPreferredSharedMemoryCarveout(const std::string& functionName, int value);

  void setKernelMaxRegisters(const std::string& functionName, int value);

  void setKernelMaxThreadsPerBlock(const std::string& functionName, int value);

  void setKernelNumRegs(const std::string& functionName, int value);

  void setKernelSharedSizeBytes(const std::string& functionName, int value);

  void setKernelBinaryVersion(const std::string& functionName, int value);

  void setKernelCacheModeCA(const std::string& functionName, int value);

  void setKernelMaxThreadsPerBlockOptIn(const std::string& functionName, int value);

  void setKernelReservedSharedSizeBytes(const std::string& functionName, int value);

  void setAllKernelsAttribute(CUfunction_attribute attribute, int value);


  void setAllKernelsMaxDynamicSharedSizeBytes(int value);
  void setAllKernelsPreferredSharedMemoryCarveout(int value);
  void setAllKernelsMaxRegisters(int value);
  void setAllKernelsMaxThreadsPerBlock(int value);
  void setAllKernelsNumRegs(int value);
  void setAllKernelsSharedSizeBytes(int value);
  void setAllKernelsBinaryVersion(int value);
  void setAllKernelsCacheModeCA(int value);
  void setAllKernelsMaxThreadsPerBlockOptIn(int value);
  void setAllKernelsReservedSharedSizeBytes(int value);

  void printKernelAttribute(const char* name, CUfunction_attribute attribute);
  void printMaxKernelAttributes();
  void printKernelAttributes(const char* name);
  std::map<std::string, ValidationResult> collectResourceProblems();
  void printValidationResult(const char* name, ValidationResult& result);
  ValidationResult validateKernelLaunch(const char* name, dim3 threadsPerBlock, dim3 numBlocks, size_t globalMemoryUsage, int minComputeCapability);

  void printProblematicFunctions();

  std::vector<std::string> parseCUBINFile(const std::string& filePath);
  std::vector<std::string> parsePTXFile(const std::string& filePath);
};

