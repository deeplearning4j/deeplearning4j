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
// Created by raver119 on 01.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_size)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(size, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(output->isScalar(), 0, "Size output should be scalar");

  output->p(0, input->lengthOf());
  output->syncToDevice();

  return Status::OK;
}
DECLARE_SHAPE_FN(size) { return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(sd::DataType::INT64)); }

DECLARE_TYPES(size) {
  getOpDescriptor()
      ->setAllowedInputTypes(ANY)
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS})
      ->allowOverride(true);
}
}  // namespace ops
}  // namespace sd

#endif
