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

PLATFORM_IMPL(log_softmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();

  const int64_t inner_dim = input->sizeAt(rank-1);
  const int64_t outer_dim = input->lengthOf() / inner_dim ;
  auto ret = vednnSoftmaxForward( VEDNN_SOFTMAX_LOG, input->buffer(), output->buffer(), outer_dim, inner_dim);

  return ret == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
}

PLATFORM_CHECK(log_softmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  Requirements req("VEDNN LOG SOFTMAX OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectIn(makeInfoVariable(dim, "The dimension would be performed on"),  {-1, rank - 1} ) &&
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
