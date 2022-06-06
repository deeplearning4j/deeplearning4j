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

PLATFORM_IMPL(permute, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  std::vector<int> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getIArguments();

  VEDAdeviceptr vIn, vO;
  vIn = (VEDAdeviceptr)input->specialBuffer();
  vO = (VEDAdeviceptr)output->specialBuffer();

  auto func = handle.getFunctionByConstPtrName("vedaPermuteAssignRank2_4");
  VEDA_CALL_THROW(vedaLaunchKernel(
      func, 0, VEDAstack((void*)input->shapeInfo(), VEDA_ARGS_INTENT_IN, shape::shapeInfoByteLength(input->rankOf())),
      vIn, VEDAstack((void*)output->shapeInfo(), VEDA_ARGS_INTENT_IN, shape::shapeInfoByteLength(output->rankOf())), vO,
      VEDAstack(permutationVector.data(), VEDA_ARGS_INTENT_IN, permutationVector.size() * sizeof(int))));

  return sd::Status::OK;
}

PLATFORM_CHECK(permute, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("VEDNN PERMUTE OP");
  size_t permutationVectorSize = block.width() > 1 ? INPUT_VARIABLE(1)->lengthOf() : block.getIArguments()->size();
  req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT), 1) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT)) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), IS_EMPTY_MSG_OUTPUT)) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(permutationVectorSize, "Permutation Vector size"), input->rankOf());
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif
