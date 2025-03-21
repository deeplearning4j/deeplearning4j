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
// Created by raver119 on 23.11.17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_toggle_bits)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/toggle_bits.h>

namespace sd {
namespace ops {
OP_IMPL(toggle_bits, -1, -1, true) {
  for (size_t i = 0; i < block.width(); i++) {
    auto x = INPUT_VARIABLE(i);
    auto z = OUTPUT_VARIABLE(i);

    REQUIRE_TRUE(x->dataType() == z->dataType(), 0, "Toggle bits requires input and output to have same type");
    REQUIRE_TRUE(x->isZ(), 0, "Toggle bits requires input and output to be integer type (int8, int16, int32, int64)");

    helpers::__toggle_bits(block.launchContext(), x, z);
  }
  return Status::OK;
}

DECLARE_TYPES(toggle_bits) {
  getOpDescriptor()->setAllowedInputTypes({ALL_INTS})->setAllowedOutputTypes({ALL_INTS})->setSameMode(false);
}
}  // namespace ops
}  // namespace sd

#endif
