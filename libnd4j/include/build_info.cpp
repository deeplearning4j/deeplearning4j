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

// Build type information at compile time
std::vector<std::string> buildSupportedTypes() {
 std::vector<std::string> types;

#if defined(HAS_BOOL)
 types.push_back("bool");
#endif
#if defined(HAS_INT8)
 types.push_back("int8");
#endif
#if defined(HAS_UINT8)
 types.push_back("uint8");
#endif
#if defined(HAS_INT16)
 types.push_back("int16");
#endif
#if defined(HAS_UINT16)
 types.push_back("uint16");
#endif
#if defined(HAS_INT32)
 types.push_back("int32");
#endif
#if defined(HAS_UINT32)
 types.push_back("uint32");
#endif
#if defined(HAS_INT64)
 types.push_back("int64");
#endif
#if defined(HAS_UINT64)
 types.push_back("uint64");
#endif
#if defined(HAS_FLOAT16)
 types.push_back("float16");
#endif
#if defined(HAS_BFLOAT16)
 types.push_back("bfloat16");
#endif
#if defined(HAS_FLOAT32)
 types.push_back("float32");
#endif
#if defined(HAS_DOUBLE)
 types.push_back("double");
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

 // NEW: Add type information to build info
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