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
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_broadcast_to)

#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(linear_copy, 2, 1, false, 0, 0) {
  auto output = OUTPUT_VARIABLE(0);
  auto input = INPUT_VARIABLE(0);

  input->applyPairwiseTransform(pairwise::CopyPws,*input, *output);
  return Status::OK;
}

DECLARE_TYPES(linear_copy) { getOpDescriptor()->setAllowedInputTypes(ANY); }

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(linear_copy) {
  auto input = INPUT_VARIABLE(0);
  auto shape = INPUT_VARIABLE(1);
  ShapeDescriptor *desc = new ShapeDescriptor(input->dataType(), shape::order(input->shapeInfo()), shape->getBufferAsVector<LongType>());
  auto outShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(desc);
  return SHAPELIST(outShapeInfo);

}

}  // namespace ops
}  // namespace sd

#endif
