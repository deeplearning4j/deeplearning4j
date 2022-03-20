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
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(relu, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
#if !defined(HAVE_VEDA)
  auto ret = vednnActivationForward(VEDNN_ACTIVATION_RELU, input->buffer(), output->buffer(), input->lengthOf());
  return ret == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else
  VEDA_HANDLE &handle = VEDA_HANDLE::getInstance();

  auto func = handle.getFunctionByConstPtrName("vedaVednnActivationForward");

  VEDAdeviceptr vIn, vO;
  size_t sizeIn = input->lengthOf() * input->sizeOfT();
  size_t sizeO = output->lengthOf() * output->sizeOfT();

  VEDA(vedaMemAllocAsync(&vIn, sizeIn, 0));
  VEDA(vedaMemAllocAsync(&vO, sizeO, 0));

  VEDA(vedaMemcpyHtoDAsync(vIn, input->buffer(), sizeIn, 0));

  const unsigned long nElements = input->lengthOf();

  VEDA(vedaLaunchKernel(func, 0, VEDNN_ACTIVATION_RELU, vIn, vO, nElements));

  VEDA(vedaMemcpyDtoHAsync(output->buffer(), vO, sizeO, 0));

  VEDA(vedaCtxSynchronize());

  VEDA(vedaMemFreeAsync(vIn, 0));
  VEDA(vedaMemFreeAsync(vO, 0));
  return sd::Status::OK;
#endif
}

PLATFORM_CHECK(relu, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto scalar = block.numT() > 0 ? block.getTArguments()->at(0) : 0.0;

  Requirements req("VEDNN RELU OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(scalar, "The Relu scalar"),  0 ) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

 
PLATFORM_IMPL(relu_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);
#if !defined(HAVE_VEDA)
  auto ret = vednnActivationBackward(VEDNN_ACTIVATION_RELU, gradO->buffer(), input->buffer(), gradI->buffer(), input->lengthOf());
  return ret == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else
  VEDA_HANDLE &handle = VEDA_HANDLE::getInstance();

  auto func = handle.getFunctionByConstPtrName("vedaVednnActivationBackward");

  VEDAdeviceptr vGradOut, vIn, vGradIn;

  VEDA(vedaMemAllocAsync(&vGradOut, gradO->lengthOf() * gradO->sizeOfT(), 0));
  VEDA(vedaMemAllocAsync(&vIn, input->lengthOf() * input->sizeOfT(), 0));
  VEDA(vedaMemAllocAsync(&vGradIn, gradI->lengthOf() * gradI->sizeOfT(), 0));
  VEDA(vedaMemcpyHtoDAsync(vGradOut, gradO->buffer(), gradO->lengthOf() * gradO->sizeOfT(), 0));
  VEDA(vedaMemcpyHtoDAsync(vIn, input->buffer(), input->lengthOf() * input->sizeOfT(), 0));

  const unsigned long nElements = input->lengthOf();

  VEDA(vedaLaunchKernel(func, 0, VEDNN_ACTIVATION_RELU, vGradOut, vIn, vGradIn, nElements));
  VEDA(vedaMemcpyDtoHAsync(gradI->buffer(), vGradIn, gradI->lengthOf() * gradI->sizeOfT(), 0));
  VEDA(vedaCtxSynchronize());

  VEDA(vedaMemFreeAsync(vGradOut, 0));
  VEDA(vedaMemFreeAsync(vIn, 0));
  VEDA(vedaMemFreeAsync(vGradIn, 0));
  return sd::Status::OK;
#endif
}

PLATFORM_CHECK(relu_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("VEDNN RELU_BP OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(gradO->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(gradO->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(gradI->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(gradO->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(gradI->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
