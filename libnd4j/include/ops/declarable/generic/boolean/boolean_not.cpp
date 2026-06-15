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
// Created by raver on 6/6/2018.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_boolean_not)

#include <ops/declarable/headers/boolean.h>

namespace sd {
namespace ops {
OP_IMPL(boolean_not, 1, 1, true) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // Use the built-in Not transform which dispatches to CUDA kernel on GPU.
  // The previous host-side loop (syncToHost + CPU loop + syncToDevice) was
  // capture-incompatible: the H2D/D2H copies invalidated CUDA graph capture,
  // preventing composite replay and dropping throughput from 60+ to 5 tok/s.
  x->applyTransform(sd::transform::BoolOps::Not, z);

  return Status::OK;
}

DECLARE_TYPES(boolean_not) {
  getOpDescriptor()->setAllowedInputTypes(0, BOOL)->setAllowedOutputTypes(0, BOOL);
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_LOGICAL);
}
}  // namespace ops
}  // namespace sd

#endif
