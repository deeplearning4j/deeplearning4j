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
// @author raver119@gmail.com, created on 29/10/17
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_softmax)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>

namespace sd {
namespace ops {

DECLARE_TYPES(softmax) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})->setSameMode(true);
}

CONFIGURABLE_OP_IMPL(softmax, 1, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  REQUIRE_TRUE(dim < rank, 0,
               "SOFTMAX OP: the value of input integer parameter (dimension) must be less than input array rank %i, "
               "but got dimension = %i instead !",
               rank, dim);

  helpers::softmax(block.launchContext(), input, output, dim);

  return sd::Status::OK;
}

CONFIGURABLE_OP_IMPL(softmax_bp, 3, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto softmaxedOut = INPUT_VARIABLE(2);
  auto gradI = OUTPUT_VARIABLE(0);
  gradI->assign(softmaxedOut);
  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  REQUIRE_TRUE(dim < rank, 0,
               "SOFTMAX_BP OP: the value of input integer parameter (dimension) must be less than input array rank %i, "
               "but got dimension = %i instead !",
               rank, dim);


  std::vector<sd::LongType> dimVector = {dim};
  auto sumAlongDim = (*gradI * *gradO).reduceAlongDimension(reduce::Sum, &dimVector, true);
  NDArray assign =  *gradI * (*gradO - sumAlongDim);
  gradI->assign(&assign);

  return sd::Status::OK;
}

DECLARE_TYPES(softmax_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
