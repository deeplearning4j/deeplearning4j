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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

#if defined(HAVE_VEDA)

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_TRANSFORM_STRICT_IMPL(Exp, ENGINE_CPU) {
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaExpF32");

  VEDAdeviceptr vIn, vO;
  const sd::LongType len = shape::length(inArg0ShapeInfo);
  // we will not use the offset here as it was not used
  vIn = (VEDAdeviceptr)inArg0Buffer->special();
  vO = (VEDAdeviceptr)outputBuffer->special();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)len, vIn, vO));

  return sd::Status::OK;
}

PLATFORM_TRANSFORM_STRICT_CHECK(Exp, ENGINE_CPU) {
  const sd::LongType xEws = shape::elementWiseStride(inArg0ShapeInfo);
  Requirements req("VEDNN Exp TrasnformStrict OP");
  req.expectEq(makeInfoVariable(ArrayOptions::dataType(inArg0ShapeInfo), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(ArrayOptions::dataType(outShapeInfo), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(xEws, EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(shape::elementWiseStride(outShapeInfo), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

PLATFORM_TRANSFORM_STRICT_IMPL(Log, ENGINE_CPU) {
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaLogF32");

  VEDAdeviceptr vIn, vO;
  const sd::LongType len = shape::length(inArg0ShapeInfo);
  // we will not use the offset here as it was not used
  vIn = (VEDAdeviceptr)inArg0Buffer->special();
  vO = (VEDAdeviceptr)outputBuffer->special();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)len, vIn, vO));
  return sd::Status::OK;
}

PLATFORM_TRANSFORM_STRICT_CHECK(Log, ENGINE_CPU) {
  const sd::LongType xEws = shape::elementWiseStride(inArg0ShapeInfo);
  Requirements req("VEDNN Log TrasnformStrict OP");
  req.expectEq(makeInfoVariable(ArrayOptions::dataType(inArg0ShapeInfo), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(ArrayOptions::dataType(outShapeInfo), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(xEws, EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(shape::elementWiseStride(outShapeInfo), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

PLATFORM_TRANSFORM_STRICT_IMPL(Tanh, ENGINE_CPU) {
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaTanhF32");

  VEDAdeviceptr vIn, vO;
  const sd::LongType len = shape::length(inArg0ShapeInfo);
  // we will not use the offset here as it was not used
  vIn = (VEDAdeviceptr)inArg0Buffer->special();
  vO = (VEDAdeviceptr)outputBuffer->special();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)len, vIn, vO));
  return sd::Status::OK;
}

PLATFORM_TRANSFORM_STRICT_CHECK(Tanh, ENGINE_CPU) {
  const sd::LongType xEws = shape::elementWiseStride(inArg0ShapeInfo);
  Requirements req("VEDNN Tanh TrasnformStrict OP");
  req.expectEq(makeInfoVariable(ArrayOptions::dataType(inArg0ShapeInfo), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(ArrayOptions::dataType(outShapeInfo), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(xEws, EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(shape::elementWiseStride(outShapeInfo), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
  ;
}

PLATFORM_TRANSFORM_STRICT_IMPL(Sigmoid, ENGINE_CPU) {
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaSigmoidF32");

  VEDAdeviceptr vIn, vO;
  const sd::LongType len = shape::length(inArg0ShapeInfo);
  // we will not use the offset here as it was not used
  vIn = (VEDAdeviceptr)inArg0Buffer->special();
  vO = (VEDAdeviceptr)outputBuffer->special();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)len, vIn, vO));
  return sd::Status::OK;
}

PLATFORM_TRANSFORM_STRICT_CHECK(Sigmoid, ENGINE_CPU) {
  const sd::LongType xEws = shape::elementWiseStride(inArg0ShapeInfo);
  Requirements req("VEDNN Sigmoid TrasnformStrict OP");
  req.expectEq(makeInfoVariable(ArrayOptions::dataType(inArg0ShapeInfo), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(ArrayOptions::dataType(outShapeInfo), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(xEws, EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(shape::elementWiseStride(outShapeInfo), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif
