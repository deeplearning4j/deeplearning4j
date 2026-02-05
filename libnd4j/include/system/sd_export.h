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

//
// Minimal export macros header to avoid circular dependencies
// This header should ONLY define SD_LIB_EXPORT and SD_LIB_HIDDEN
// with no other includes to prevent circular dependency issues.
//

#ifndef SD_EXPORT_H
#define SD_EXPORT_H

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define SD_LIB_EXPORT __attribute__((dllexport))
#else
#define SD_LIB_EXPORT __declspec(dllexport)
#endif
#define SD_LIB_HIDDEN
#else
#if __GNUC__ >= 4
#define SD_LIB_EXPORT __attribute__((visibility("default")))
#define SD_LIB_HIDDEN __attribute__((visibility("hidden")))
#else
#define SD_LIB_EXPORT
#define SD_LIB_HIDDEN
#endif
#endif

#endif // SD_EXPORT_H
