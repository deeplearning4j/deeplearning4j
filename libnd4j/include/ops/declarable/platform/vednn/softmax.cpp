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

PLATFORM_IMPL(softmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();

  const unsigned long inner_dim = input->sizeAt(rank - 1);
  const unsigned long outer_dim = input->lengthOf() / inner_dim;

#if !defined(HAVE_VEDA)
  auto ret = vednnSoftmaxForward(VEDNN_SOFTMAX_ACCURATE, input->buffer(), output->buffer(), outer_dim, inner_dim);
  return ret == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else
  VEDA_HANDLE &handle = VEDA::getInstance().getVEDA_HANDLE(0);
  SCOPED_VEDA_CONTEXT scopeContext(handle.getDevice());

  auto func = handle.getFunctionByConstPtrName("vedaVednnSoftmaxForward");

  VEDAdeviceptr vIn, vO;
  size_t sizeIn = input->lengthOf() * input->sizeOfT();
  size_t sizeO = output->lengthOf() * output->sizeOfT();

  VEDA_CALL_THROW(vedaMemAllocAsync(&vIn, sizeIn, 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vO, sizeO, 0));

  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vIn, input->buffer(), sizeIn, 0));

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, VEDNN_SOFTMAX_ACCURATE, vIn, vO, outer_dim, inner_dim));

  VEDA_CALL_THROW(vedaMemcpyDtoHAsync(output->buffer(), vO, sizeO, 0));

  VEDA_CALL_THROW(vedaCtxSynchronize());

  VEDA_CALL_THROW(vedaMemFreeAsync(vIn, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vO, 0));
  return sd::Status::OK;
#endif
}

PLATFORM_CHECK(softmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  Requirements req("VEDNN SOFTMAX OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectIn(makeInfoVariable(dim, "The dimension would be performed on"), {-1, rank - 1}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
