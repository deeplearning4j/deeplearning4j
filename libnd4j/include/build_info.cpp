/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#include <build_info.h>
#include <config.h>

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cctype>

#include "helpers/logger.h"

#if defined(SD_GCC_FUNCTRACE)

bool isFuncTrace() {
  return true;
}

#else

bool isFuncTrace() {
  return false;
}

#endif

// Helper function to convert string to lowercase
std::string toLowerCase(const std::string& str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

// Helper function to convert type name to build script format
std::string typeToScriptFormat(const std::string& typeName) {
  std::string lower = toLowerCase(typeName);

  // Convert runtime type names to build script format
  if (lower == "int64" || lower == "int64_t" || lower == "long") return "int64";
  if (lower == "int32" || lower == "int32_t" || lower == "int") return "int32";
  if (lower == "int16" || lower == "int16_t") return "int16";
  if (lower == "int8" || lower == "int8_t") return "int8";
  if (lower == "uint64" || lower == "uint64_t" || lower == "unsignedlong") return "uint64";
  if (lower == "uint32" || lower == "uint32_t") return "uint32";
  if (lower == "uint16" || lower == "uint16_t") return "uint16";
  if (lower == "uint8" || lower == "uint8_t") return "uint8";
  if (lower == "float32" || lower == "float") return "float32";
  if (lower == "double" || lower == "float64") return "double";
  if (lower == "float16" || lower == "half") return "float16";
  if (lower == "bfloat16" || lower == "bfloat") return "bfloat16";
  if (lower == "bool") return "bool";

  // String types
  if (lower == "std::string" || lower == "utf8") return "utf8";
  if (lower == "std::u16string" || lower == "utf16") return "utf16";
  if (lower == "std::u32string" || lower == "utf32") return "utf32";

  // Fallback - return as-is
  return lower;
}
// Build type information at compile time - FIXED VERSION
std::vector<std::string> buildSupportedTypes() {
  std::vector<std::string> types;

  // Boolean type
#if defined(HAS_BOOL)
  types.push_back("bool");
#endif

  // Integer types - check both normalized and alias forms
#if defined(HAS_INT8_T) || defined(HAS_INT8)
  types.push_back("int8");
#endif
#if defined(HAS_UINT8_T) || defined(HAS_UINT8)
  types.push_back("uint8");
#endif
#if defined(HAS_INT16_T) || defined(HAS_INT16)
  types.push_back("int16");
#endif
#if defined(HAS_UINT16_T) || defined(HAS_UINT16)
  types.push_back("uint16");
#endif
#if defined(HAS_INT32_T) || defined(HAS_INT32) || defined(HAS_INT)
  types.push_back("int32");
#endif
#if defined(HAS_UINT32_T) || defined(HAS_UINT32)
  types.push_back("uint32");
#endif
#if defined(HAS_INT64_T) || defined(HAS_INT64) || defined(HAS_LONG)
  types.push_back("int64");
#endif
#if defined(HAS_UINT64_T) || defined(HAS_UINT64) || defined(HAS_UNSIGNEDLONG)
  types.push_back("uint64");
#endif

  // Floating point types - check both normalized and alias forms
#if defined(HAS_FLOAT) || defined(HAS_FLOAT32)
  types.push_back("float32");
#endif
#if defined(HAS_DOUBLE) || defined(HAS_FLOAT64)
  types.push_back("double");
#endif
#if defined(HAS_FLOAT16) || defined(HAS_HALF)
  types.push_back("float16");
#endif
#if defined(HAS_BFLOAT16) || defined(HAS_BFLOAT)
  types.push_back("bfloat16");
#endif

  // String types - check if string operations are enabled
#if defined(SD_ENABLE_STRING_OPERATIONS)
#if defined(HAS_STD_STRING)
  types.push_back("utf8");
#endif
#if defined(HAS_STD_U16STRING)
  types.push_back("utf16");
#endif
#if defined(HAS_STD_U32STRING)
  types.push_back("utf32");
#endif
#endif

  // Sort types for consistent output
  std::sort(types.begin(), types.end());
  return types;
}


// Static storage for type information - initialized once
static std::vector<std::string> supportedTypes = buildSupportedTypes();
static std::string supportedTypesString;
static std::string binaryTypeInfoString;
static std::vector<const char*> supportedTypesArray;
static bool typesInitialized = false;

void initializeTypeInfo() {
  if (typesInitialized) return;

  // Build comma-separated string
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    if (i > 0) supportedTypesString += ",";
    supportedTypesString += supportedTypes[i];
  }

  // Build C-style array
  supportedTypesArray.clear();
  for (const auto& type : supportedTypes) {
    supportedTypesArray.push_back(type.c_str());
  }
  supportedTypesArray.push_back(nullptr); // Null terminate

  // Build comprehensive type info string
  binaryTypeInfoString = "Binary Type Configuration:\n";
  binaryTypeInfoString += "Selection Mode: ";
#if defined(SD_SELECTIVE_TYPES)
  binaryTypeInfoString += "SELECTIVE\n";
#else
  binaryTypeInfoString += "ALL_TYPES\n";
#endif
  binaryTypeInfoString += "Supported Types (" + std::to_string(supportedTypes.size()) + "): ";
  binaryTypeInfoString += supportedTypesString + "\n";

  // Add categorization
  std::vector<std::string> integerTypes, floatTypes, boolTypes;
  for (const auto& type : supportedTypes) {
    if (type == "bool") {
      boolTypes.push_back(type);
    } else if (type.find("int") != std::string::npos || type.find("uint") != std::string::npos) {
      integerTypes.push_back(type);
    } else if (type.find("float") != std::string::npos || type == "double" || type.find("bfloat") != std::string::npos) {
      floatTypes.push_back(type);
    }
  }

  if (!boolTypes.empty()) {
    binaryTypeInfoString += "Boolean Types: ";
    for (size_t i = 0; i < boolTypes.size(); ++i) {
      if (i > 0) binaryTypeInfoString += ",";
      binaryTypeInfoString += boolTypes[i];
    }
    binaryTypeInfoString += "\n";
  }

  if (!integerTypes.empty()) {
    binaryTypeInfoString += "Integer Types: ";
    for (size_t i = 0; i < integerTypes.size(); ++i) {
      if (i > 0) binaryTypeInfoString += ",";
      binaryTypeInfoString += integerTypes[i];
    }
    binaryTypeInfoString += "\n";
  }

  if (!floatTypes.empty()) {
    binaryTypeInfoString += "Floating Point Types: ";
    for (size_t i = 0; i < floatTypes.size(); ++i) {
      if (i > 0) binaryTypeInfoString += ",";
      binaryTypeInfoString += floatTypes[i];
    }
    binaryTypeInfoString += "\n";
  }

  typesInitialized = true;
}
std::string typeToCMakeFormat(const std::string& typeName) {
  std::string lower = toLowerCase(typeName);

  // Convert script type names to CMake format (what CMake uses internally)
  if (lower == "int64" || lower == "long") return "int64_t";
  if (lower == "int32" || lower == "int") return "int32_t";
  if (lower == "int16") return "int16_t";
  if (lower == "int8") return "int8_t";
  if (lower == "uint64" || lower == "unsignedlong") return "uint64_t";
  if (lower == "uint32") return "uint32_t";
  if (lower == "uint16") return "uint16_t";
  if (lower == "uint8") return "uint8_t";
  if (lower == "float32") return "float";
  if (lower == "float64") return "double";
  if (lower == "float16" || lower == "half") return "float16";
  if (lower == "bfloat16" || lower == "bfloat") return "bfloat16";
  if (lower == "bool") return "bool";

  // String types
  if (lower == "utf8") return "std::string";
  if (lower == "utf16") return "std::u16string";
  if (lower == "utf32") return "std::u32string";

  // Fallback
  return lower;
}


// CRITICAL FIX: Generate proper error message for missing types
std::string generateMissingTypeError(const std::string& missingType, const std::string& context) {
  initializeTypeInfo(); // Ensure type info is initialized

  std::ostringstream message;

  // Header with clear error indication
  message << "❌ ERROR: Data type '" << missingType << "' is not available in this build\n";

  if (!context.empty()) {
    message << "Context: " << context << "\n";
  }

  message << "Build configuration: ";
#if defined(SD_SELECTIVE_TYPES)
  message << "Selective types enabled (only specified types available)\n";
#else
  message << "All types enabled (this should not happen)\n";
#endif

  // Show available types in this build
  message << "Available types in this build (" << supportedTypes.size() << "): ";
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    if (i > 0) message << ", ";
    message << supportedTypes[i];
  }
  message << "\n\n";

  // FIXED: Provide correct rebuild instructions based on current build mode
  message << "To enable this type, rebuild with:\n\n";

  // Build script format - include current types + missing type
  std::string scriptType = typeToScriptFormat(missingType);
  message << "Build Script:\n";
  message << "  ./buildnativeoperations.sh --datatypes \"";

  // Include current types + missing type
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    message << supportedTypes[i] << ";";
  }
  message << scriptType << "\"\n\n";

  // Maven format
  message << "Maven:\n";
  message << "  mvn compile -Dlibnd4j.datatypes=\"";
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    message << supportedTypes[i] << ";";
  }
  message << scriptType << "\"\n\n";

  // CMake format - use normalized CMake type names
  message << "CMake:\n";
  message << "  cmake -DSD_TYPES_LIST=\"";
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    message << typeToCMakeFormat(supportedTypes[i]) << ";";
  }
  message << typeToCMakeFormat(scriptType) << "\" ..\n\n";

  // Suggest type profiles for common cases
  message << "Or use a predefined type profile:\n";
  message << "  ./buildnativeoperations.sh --debug-type-profile ESSENTIAL\n";
  message << "  ./buildnativeoperations.sh --debug-type-profile MINIMAL_INDEXING\n";
  message << "  ./buildnativeoperations.sh --debug-type-profile QUANTIZATION\n";
  message << "  ./buildnativeoperations.sh --debug-type-profile MIXED_PRECISION\n\n";

  // Show current mode help
#if defined(SD_SELECTIVE_TYPES)
  message << "Current mode: SELECTIVE TYPES\n";
  message << "This build was configured with specific data types only.\n";
  message << "To include all types, build without --datatypes parameter.\n\n";
#else
  message << "Current mode: ALL TYPES\n";
  message << "This should not happen - all types should be available.\n";
  message << "This may indicate a build system bug.\n\n";
#endif

  // Additional help
  message << "To see all available types and profiles:\n";
  message << "  ./buildnativeoperations.sh --show-types\n";
  message << "  ./buildnativeoperations.sh --help\n";

  return message.str();
}

// NEW: Generate type mismatch error for operations
std::string generateTypeMismatchError(const std::string& operation,
                                      const std::vector<std::string>& requiredTypes,
                                      const std::string& context) {
  initializeTypeInfo();

  std::ostringstream message;

  message << "❌ ERROR: Operation '" << operation << "' requires types that are not available\n";

  if (!context.empty()) {
    message << "Context: " << context << "\n";
  }

  message << "Required types: ";
  for (size_t i = 0; i < requiredTypes.size(); ++i) {
    if (i > 0) message << ", ";
    message << requiredTypes[i];
  }
  message << "\n";

  message << "Available types: ";
  for (size_t i = 0; i < supportedTypes.size(); ++i) {
    if (i > 0) message << ", ";
    message << supportedTypes[i];
  }
  message << "\n\n";

  // Find missing types
  std::vector<std::string> missingTypes;
  for (const auto& reqType : requiredTypes) {
    bool found = false;
    for (const auto& availType : supportedTypes) {
      if (toLowerCase(reqType) == toLowerCase(availType)) {
        found = true;
        break;
      }
    }
    if (!found) {
      missingTypes.push_back(reqType);
    }
  }

  if (!missingTypes.empty()) {
    message << "Missing types: ";
    for (size_t i = 0; i < missingTypes.size(); ++i) {
      if (i > 0) message << ", ";
      message << missingTypes[i];
    }
    message << "\n\n";

    message << "To fix this, rebuild with all required types:\n";
    message << "  ./buildnativeoperations.sh --datatypes \"";

    // Combine current + missing types
    std::vector<std::string> allNeededTypes = supportedTypes;
    for (const auto& missing : missingTypes) {
      allNeededTypes.push_back(typeToScriptFormat(missing));
    }

    for (size_t i = 0; i < allNeededTypes.size(); ++i) {
      if (i > 0) message << ";";
      message << allNeededTypes[i];
    }
    message << "\"\n";
  }

  return message.str();
}

// NEW: Get detailed type configuration info for debugging
std::string getDetailedTypeConfiguration() {
  initializeTypeInfo();

  std::ostringstream info;

  info << "=== DETAILED TYPE CONFIGURATION ===\n";
  info << "Selection Mode: " << getTypeSelectionMode() << "\n";
  info << "Total Types: " << supportedTypes.size() << "\n\n";

  // Categorize types
  std::vector<std::string> integerTypes, floatTypes, boolTypes, otherTypes;

  for (const auto& type : supportedTypes) {
    if (type == "bool") {
      boolTypes.push_back(type);
    } else if (type.find("int") != std::string::npos || type.find("uint") != std::string::npos) {
      integerTypes.push_back(type);
    } else if (type.find("float") != std::string::npos || type == "double" || type.find("bfloat") != std::string::npos) {
      floatTypes.push_back(type);
    } else {
      otherTypes.push_back(type);
    }
  }

  if (!boolTypes.empty()) {
    info << "Boolean Types (" << boolTypes.size() << "): ";
    for (size_t i = 0; i < boolTypes.size(); ++i) {
      if (i > 0) info << ", ";
      info << boolTypes[i];
    }
    info << "\n";
  }

  if (!integerTypes.empty()) {
    info << "Integer Types (" << integerTypes.size() << "): ";
    for (size_t i = 0; i < integerTypes.size(); ++i) {
      if (i > 0) info << ", ";
      info << integerTypes[i];
    }
    info << "\n";
  }

  if (!floatTypes.empty()) {
    info << "Floating Point Types (" << floatTypes.size() << "): ";
    for (size_t i = 0; i < floatTypes.size(); ++i) {
      if (i > 0) info << ", ";
      info << floatTypes[i];
    }
    info << "\n";
  }

  if (!otherTypes.empty()) {
    info << "Other Types (" << otherTypes.size() << "): ";
    for (size_t i = 0; i < otherTypes.size(); ++i) {
      if (i > 0) info << ", ";
      info << otherTypes[i];
    }
    info << "\n";
  }

  info << "\n";

  // Build configuration hints
  info << "Build Configuration:\n";
#if defined(SD_SELECTIVE_TYPES)
  info << "  - Selective types are ENABLED\n";
  info << "  - Only specified types are available\n";
  info << "  - Rebuild with additional types if needed\n";
#else
  info << "  - All types are ENABLED\n";
  info << "  - Complete type support available\n";
#endif

#if defined(SD_GCC_FUNCTRACE)
  info << "  - Function tracing is ENABLED\n";
  info << "  - Debug build configuration detected\n";
#endif

  info << "\n=== END TYPE CONFIGURATION ===\n";

  return info.str();
}
// Type information API implementation
const char *getSupportedTypes() {
  initializeTypeInfo();
  return supportedTypesString.c_str();
}

const char **getSupportedTypesArray() {
  initializeTypeInfo();
  return supportedTypesArray.data();
}

int getSupportedTypesCount() {
  return static_cast<int>(supportedTypes.size());
}

bool isTypeSupported(const char* typeName) {
  if (!typeName) return false;
  initializeTypeInfo();

  std::string typeStr(typeName);
  for (const auto& type : supportedTypes) {
    if (type == typeStr) return true;
  }
  return false;
}

bool hasSelectiveTypes() {
#if defined(SD_SELECTIVE_TYPES)
  return true;
#else
  return false;
#endif
}

const char *getTypeSelectionMode() {
#if defined(SD_SELECTIVE_TYPES)
  return "SELECTIVE";
#else
  return "ALL_TYPES";
#endif
}

const char *getBinaryTypeInfo() {
  initializeTypeInfo();
  return binaryTypeInfoString.c_str();
}

// NEW: Get missing type error message (for runtime use)
const char *getMissingTypeError(const char* missingType, const char* context) {
  static std::string errorMessage;
  std::string contextStr = context ? std::string(context) : std::string("");
  errorMessage = generateMissingTypeError(missingType ? std::string(missingType) : "UNKNOWN", contextStr);
  return errorMessage.c_str();
}

// NEW: Get detailed type configuration (for debugging)
// NEW: Get detailed type configuration (for debugging)
const char *getDetailedTypeConfig() {
  static std::string detailedConfig;
  detailedConfig = getDetailedTypeConfiguration();
  return detailedConfig.c_str();
}

const char *buildInfo() {
  static std::string ret;
  if (!ret.empty()) return ret.c_str(); // Cache the result

  ret = "Build Info:\n";

#if defined(SD_GCC_FUNCTRACE)
  ret += "Functrace: ";
  ret += isFuncTrace() ? "ON\n" : "OFF\n";
#endif

#if defined(__clang__)
  ret += "Clang: " STRINGIZE(__clang_version__) "\n";
#elif defined(_MSC_VER)
  ret += "MSVC: " STRINGIZE(_MSC_FULL_VER) "\n";
#else
  ret += "GCC: " STRINGIZE(__VERSION__) "\n";
#endif

#if defined(_MSC_VER) && defined(_MSVC_LANG)
  ret += "STD version: " STRINGIZE(_MSVC_LANG) "\n";
#elif defined(__cplusplus)
  ret += "STD version: " STRINGIZE(__cplusplus) "\n";
#endif

#if defined(__CUDACC__)
  ret += "CUDA: " STRINGIZE(__CUDACC_VER_MAJOR__) "." STRINGIZE(__CUDACC_VER_MINOR__) "." STRINGIZE(__CUDACC_VER_BUILD__) "\n";
#endif

#if defined(SD_CUDA)
  ret += "CUDA: " STRINGIZE(__CUDACC_VER_MAJOR__) "." STRINGIZE(__CUDACC_VER_MINOR__) "." STRINGIZE(__CUDACC_VER_BUILD__) "\n";
#endif

#if defined(CUDA_ARCHITECTURES)
  ret += "CUDA_ARCHITECTURES: " STRINGIZE(CUDA_ARCHITECTURES) "\n";
#endif

#if defined(DEFAULT_ENGINE)
  ret += "DEFAULT_ENGINE: " STRINGIZE(DEFAULT_ENGINE) "\n";
#endif

#if defined(HAVE_FLATBUFFERS)
  ret += "HAVE_FLATBUFFERS\n";
#endif

#if defined(HAVE_ONEDNN)
  ret += "HAVE_ONEDNN\n";
#endif

#if defined(__EXTERNAL_BLAS__)
  ret += "HAVE_EXTERNAL_BLAS\n";
#endif

#if defined(HAVE_OPENBLAS)
  ret += "HAVE_OPENBLAS\n";
#endif

#if defined(HAVE_CUDNN)
  ret += "HAVE_CUDNN\n";
#endif

#if defined(HAVE_ARMCOMPUTE)
  ret += "HAVE_ARMCOMPUTE\n";
#endif

  // Add type information to build info
  initializeTypeInfo();
  ret += "Type Selection: ";
#if defined(SD_SELECTIVE_TYPES)
  ret += "SELECTIVE\n";
#else
  ret += "ALL_TYPES\n";
#endif
  ret += "Supported Types (" + std::to_string(supportedTypes.size()) + "): ";
  ret += supportedTypesString + "\n";

#if defined(SD_PRINT_MATH)
  ret += "Use PRINT_MATH_FUNCTION_NAME to specify which math function you want to track.\n";
#endif

  // Risk of build information not being printed during debug settings
  if(isFuncTrace())
    sd_printf("%s", ret.c_str());
  return ret.c_str();
}
