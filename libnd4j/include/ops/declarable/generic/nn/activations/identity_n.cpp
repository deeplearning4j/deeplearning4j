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
//  @author sgazeos@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_identity_n)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(identity_n, 1, 1, true, 0, 0) {
  if (!block.isInplace()) {
    for (size_t i = 0; i < block.width(); ++i) {
      auto x = INPUT_VARIABLE(i);
      auto z = OUTPUT_VARIABLE(i);

      x->applyTransform(transform::Identity, z);
    }
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(identity_n) {
  auto shapes = SHAPELIST();
  for (int i = 0; i < inputShape->size(); ++i) {
    shapes->push_back(CONSTANT(inputShape->at(i)));
  }
  return shapes;
}

DECLARE_TYPES(identity_n) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
}

}  // namespace ops
}  // namespace sd

#endif
