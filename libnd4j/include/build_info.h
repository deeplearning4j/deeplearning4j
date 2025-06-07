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

// New type information functions
SD_LIB_EXPORT const char *getSupportedTypes();
SD_LIB_EXPORT const char **getSupportedTypesArray();
SD_LIB_EXPORT int getSupportedTypesCount();
SD_LIB_EXPORT bool isTypeSupported(const char* typeName);

// Additional type metadata functions
SD_LIB_EXPORT bool hasSelectiveTypes();
SD_LIB_EXPORT const char *getTypeSelectionMode();
SD_LIB_EXPORT const char *getBinaryTypeInfo();

#ifdef __cplusplus
}
#endif

#endif