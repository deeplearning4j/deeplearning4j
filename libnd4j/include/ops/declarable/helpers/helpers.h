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
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_OPS_HELPERS_H
#define LIBND4J_OPS_HELPERS_H

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <execution/LaunchContext.h>
#include <helpers/shape.h>
#include <system/op_boilerplate.h>
#include <types/float16.h>
#include <types/types.h>

#include <array>
#include <vector>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helpers/DebugHelper.h>
#include <stdio.h>
#include <stdlib.h>

#endif  // CUDACC

#endif  // LIBND4J_HELPERS_H
