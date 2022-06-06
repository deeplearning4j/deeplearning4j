/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef DEV_TESTSVEDNNUTILS_H
#define DEV_TESTSVEDNNUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <vednn.h>
#if defined(HAVE_VEDA)
#include <system/op_enums.h>

#include "veda_helper.h"
#endif
using namespace samediff;

namespace sd {
namespace ops {
namespace platforms {

/**
 * forward, backward
 */
DECLARE_PLATFORM(relu, ENGINE_CPU);
DECLARE_PLATFORM(relu_bp, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);
DECLARE_PLATFORM(conv2d, ENGINE_CPU);
DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

// only forward
DECLARE_PLATFORM(matmul, ENGINE_CPU);
DECLARE_PLATFORM(softmax, ENGINE_CPU);
DECLARE_PLATFORM(log_softmax, ENGINE_CPU);

#if defined(HAVE_VEDA)
DECLARE_PLATFORM(concat, ENGINE_CPU);
DECLARE_PLATFORM(add, ENGINE_CPU);
DECLARE_PLATFORM(multiply, ENGINE_CPU);
DECLARE_PLATFORM(permute, ENGINE_CPU);
DECLARE_PLATFORM(pad, ENGINE_CPU);

DECLARE_PLATFORM_TRANSFORM_STRICT(Exp, ENGINE_CPU);
DECLARE_PLATFORM_TRANSFORM_STRICT(Log, ENGINE_CPU);
DECLARE_PLATFORM_TRANSFORM_STRICT(Tanh, ENGINE_CPU);
DECLARE_PLATFORM_TRANSFORM_STRICT(Sigmoid, ENGINE_CPU);

DECLARE_PLATFORM_SCALAR_OP(LeakyRELU, ENGINE_CPU);
#endif

SD_INLINE vednnTensorParam_t getTensorFormat(const NDArray &in, bool isNCHW = true) {
  vednnTensorParam_t param;
  param.dtype = DTYPE_FLOAT;
  if (isNCHW) {
    param.batch = (int)in.sizeAt(0);
    param.channel = (int)in.sizeAt(1);
    param.height = (int)in.sizeAt(2);
    param.width = (int)in.sizeAt(3);
  } else {
    param.batch = (int)in.sizeAt(0);
    param.channel = (int)in.sizeAt(3);
    param.height = (int)in.sizeAt(1);
    param.width = (int)in.sizeAt(2);
  }
  return param;
}

SD_INLINE vednnFilterParam_t getFilterParam(const NDArray &weights, int wFormat) {
  //// 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
  vednnFilterParam_t paramFilter;
  paramFilter.dtype = DTYPE_FLOAT;
  if (wFormat == 0) {
    paramFilter.height = (int)weights.sizeAt(0);
    paramFilter.width = (int)weights.sizeAt(1);
    paramFilter.inChannel = (int)weights.sizeAt(2);
    paramFilter.outChannel = (int)weights.sizeAt(3);
  } else if (wFormat == 1) {
    paramFilter.outChannel = (int)weights.sizeAt(0);
    paramFilter.inChannel = (int)weights.sizeAt(1);
    paramFilter.height = (int)weights.sizeAt(2);
    paramFilter.width = (int)weights.sizeAt(3);
  } else {
    paramFilter.outChannel = (int)weights.sizeAt(0);
    paramFilter.height = (int)weights.sizeAt(1);
    paramFilter.width = (int)weights.sizeAt(2);
    paramFilter.inChannel = (int)weights.sizeAt(3);
  }
  return paramFilter;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTSVEDNNUTILS_H
