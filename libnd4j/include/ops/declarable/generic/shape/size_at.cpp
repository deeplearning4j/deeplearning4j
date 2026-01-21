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
// Created by raver119 on 12.02.18.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_size_at)

#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(size_at, 1, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  auto dim = INT_ARG(0);
  if (dim < 0) dim += rank;

  REQUIRE_TRUE(dim < rank, 0, "Size_At: Dim can't be higher then input rank")

  // Get the size at the specified dimension and write to output
  // Use p() method which handles host/device sync properly
  output->p(0, input->sizeAt(dim));

  return Status::OK;
}

DECLARE_SHAPE_FN(size_at) { return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(sd::DataType::INT64)); }

DECLARE_TYPES(size_at) {
  getOpDescriptor()
      ->setAllowedInputTypes(ANY)
      ->setAllowedOutputTypes(INT64)
      ->allowOverride(true);
}
}  // namespace ops
}  // namespace sd

#endif
