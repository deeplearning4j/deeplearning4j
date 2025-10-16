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
// Created by GS <sgazeos@gmail.com> 31.01.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_zero_fraction)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(zero_fraction, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(output->isScalar(), 0, "Rank output should be scalar");

  if (input->isEmpty()) {
    output->p<double>(0, std::numeric_limits<double>::quiet_NaN());
    return Status::OK;
  }


  auto countZero = input->reduceNumber(reduce::CountZero);
  output->p<double>(0, countZero->e<LongType>(0) / double(input->lengthOf()));
  delete countZero;
  return Status::OK;
}
DECLARE_SHAPE_FN(zero_fraction) {
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(sd::DataType::DOUBLE));
}

DECLARE_TYPES(zero_fraction) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
