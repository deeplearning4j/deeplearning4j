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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_to_double)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(to_double, 1, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (!block.isInplace()) output->assign(input);

  STORE_RESULT(output);

  return Status::OK;
}

DECLARE_TYPES(to_double) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(DOUBLE);
}

DECLARE_SHAPE_FN(to_double) {
  auto outShape = ShapeBuilders::copyShapeInfoAndType(inputShape->at(0), DOUBLE, true, block.workspace());
  return SHAPELIST(CONSTANT(outShape));
}

}  // namespace ops
}  // namespace sd

#endif
