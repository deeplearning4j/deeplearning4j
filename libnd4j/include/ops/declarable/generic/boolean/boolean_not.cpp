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

  // Sync input to host for reliable access
  x->syncToHost();

  // Direct buffer access to avoid O(n^2) sync from per-element p()/e()
  auto xBuf = x->bufferAsT<int8_t>();
  auto zBuf = z->bufferAsT<int8_t>();
  auto len = x->lengthOf();

  for (sd::LongType i = 0; i < len; i++) {
    zBuf[i] = (xBuf[i] != 0) ? 0 : 1;
  }

  // Mark host written and sync to device ONCE
  z->tickWriteHost();
  z->syncToDevice();

  return Status::OK;
}

DECLARE_TYPES(boolean_not) {
  getOpDescriptor()->setAllowedInputTypes(0, BOOL)->setAllowedOutputTypes(0, BOOL);
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_LOGICAL);
}
}  // namespace ops
}  // namespace sd

#endif
