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

#ifndef LIBND4J_BUILD_INFO_H
#define LIBND4J_BUILD_INFO_H

#include <system/common.h>

#ifdef __cplusplus
extern "C" {
#endif

// Existing build info functions
SD_LIB_EXPORT const char *buildInfo();
SD_LIB_EXPORT bool isFuncTrace();

// Type information functions
SD_LIB_EXPORT const char *getSupportedTypes();
SD_LIB_EXPORT const char **getSupportedTypesArray();
SD_LIB_EXPORT int getSupportedTypesCount();
SD_LIB_EXPORT bool isTypeSupported(const char* typeName);

// Type metadata functions
SD_LIB_EXPORT bool hasSelectiveTypes();
SD_LIB_EXPORT const char *getTypeSelectionMode();
SD_LIB_EXPORT const char *getBinaryTypeInfo();

SD_LIB_EXPORT const char *getMissingTypeError(const char* missingType, const char* context);
SD_LIB_EXPORT const char *getDetailedTypeConfig();

#ifdef __cplusplus
}

// C++ only functions for generating error messages
#include <string>
#include <vector>

// Generate proper error message for missing types
std::string generateMissingTypeError(const std::string& missingType, const std::string& context = "");

// Generate error message for operations with missing types
// Generate error message for operations with missing types
std::string generateTypeMismatchError(const std::string& operation,
                                      const std::vector<std::string>& requiredTypes,
                                      const std::string& context = "");
// Get detailed type configuration for debugging
std::string getDetailedTypeConfiguration();

// Helper functions
std::string toLowerCase(const std::string& str);
std::string typeToScriptFormat(const std::string& typeName);

#endif

#endif