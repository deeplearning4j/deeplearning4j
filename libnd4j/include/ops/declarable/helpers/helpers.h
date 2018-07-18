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
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_OPS_HELPERS_H
#define LIBND4J_OPS_HELPERS_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <graph/LaunchContext.h>
#include <types/float16.h>
#include <helpers/shape.h>
#include <NDArray.h>
#include <vector>
#include <array>
#include <Status.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#endif // CUDACC

#endif // LIBND4J_HELPERS_H
