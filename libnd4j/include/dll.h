/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

//
// Created by agibsonccc on 3/5/16.
//

#ifndef NATIVEOPERATIONS_DLL_H
#define NATIVEOPERATIONS_DLL_H
#ifdef _WIN32
//#include <windows.h>
#  define ND4J_EXPORT __declspec(dllexport)
#else
#  define ND4J_EXPORT
#endif
#endif //NATIVEOPERATIONS_DLL_H
