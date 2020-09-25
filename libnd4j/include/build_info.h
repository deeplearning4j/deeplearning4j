/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#ifdef  _WIN32
#define ND4J_EXPORT   __declspec( dllexport )
#else
#define ND4J_EXPORT
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifdef __cplusplus
extern "C" {
#endif

ND4J_EXPORT const char* buildInfo();

#ifdef __cplusplus
}
#endif

#endif
