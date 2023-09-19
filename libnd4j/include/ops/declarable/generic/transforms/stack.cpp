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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_stack)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/stack.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(stack, -1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dim < 0) dim += input->rankOf() + 1;

  // no-op in case of empty output array
  if (output->isEmpty()) return sd::Status::OK;

  // input validation
  // check whether shapes of all input array are the same
  for (sd::LongType i = 0; i < block.width() - 1; ++i)
  REQUIRE_TRUE(shape::equalsSoft((INPUT_VARIABLE(i))->shapeInfo(), (INPUT_VARIABLE(i + 1))->shapeInfo()), 0,
               "STACK op: the shapes of all input arrays must be the same !");

  REQUIRE_TRUE(
      dim <= input->rankOf(), 0,
      "STACK op: the input dimension parameter must be <= rank of input arrays shapes (rank=%i), but got %i instead !",
      input->shapeOf(), dim);

  std::vector<const NDArray*> inArrs(block.width());
  for (int i = 0; i < block.width(); ++i) inArrs[i] = INPUT_VARIABLE(i);

  //empty arrays are a no op
  if(block.width() >= 1 && !inArrs[0]->isEmpty())
    helpers::stack(block.launchContext(), inArrs, *output, dim);

  return sd::Status::OK;
}
DECLARE_SYN(pack, stack);
DECLARE_SYN(Pack, stack);

DECLARE_TYPES(stack) {
  getOpDescriptor()->setAllowedInputTypes(DataType::ANY)->setAllowedOutputTypes(DataType::ANY);
}

DECLARE_SHAPE_FN(stack) {
  sd_print("Stack shape\n");
  // check whether input dimension is within rank range
  auto inShapeInfo = inputShape->at(0);
  shape::printShapeInfo(inShapeInfo);

  int rank = shape::rank(inShapeInfo);
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dim < 0) dim += rank + 1;

  REQUIRE_TRUE(
      dim <= inShapeInfo[0], 0,
      "STACK op: the input dimension parameter must be <= rank of input arrays shapes (rank=%i), but got %i instead !",
      inShapeInfo[0], dim);


  // the rank of output ShapeInfo is larger by one compared to input ShapeInfo
  std::vector<sd::LongType> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);

  // insert (int) block.width() at dim position of input shape to get output shape
  outShape.insert(outShape.begin() + sd::LongType(dim), (sd::LongType)block.width());
  auto desc = new ShapeDescriptor(ArrayOptions::dataType(inShapeInfo), shape::order(inShapeInfo), outShape);
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
  delete desc;
  return ret;
}

}  // namespace ops
}  // namespace sd

#endif
