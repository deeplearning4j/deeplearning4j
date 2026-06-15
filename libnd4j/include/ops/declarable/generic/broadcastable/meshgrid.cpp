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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.02.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_meshgrid)

#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/helpers/meshgrid.h>

#include <numeric>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(meshgrid, -1, -1, false, 0, 0) {
  int rank = block.width();

  if (rank == 1) {
    OUTPUT_VARIABLE(0)->assign(INPUT_VARIABLE(0));
    return Status::OK;
  }

  bool swapFirst2Dims = block.getIArguments()->size() > 0 ? (bool)INT_ARG(0) : true;

  std::vector<NDArray*> inArrs(rank);
  std::vector<NDArray*> outArrs(rank);

  for (int i = 0; i < rank; ++i) {
    inArrs[i] = INPUT_VARIABLE(i);
    outArrs[i] = OUTPUT_VARIABLE(i);
  }

  helpers::meshgrid(block.launchContext(), inArrs, outArrs, swapFirst2Dims);

  return Status::OK;
}

DECLARE_TYPES(meshgrid) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(INHERIT)->setSameMode(true);
}

DECLARE_SHAPE_FN(meshgrid) {
  bool swapFirst2Dims = block.getIArguments()->size() > 0 ? (bool)INT_ARG(0) : true;

  int rank = block.width();
  
  // For meshgrid with 'xy' indexing (swapFirst2Dims=true):
  // - output[0] has shape [len(input[1]), len(input[0]), 1, 1, ...]
  // - output[1] has shape [len(input[1]), len(input[0]), 1, 1, ...]
  // - output[i] for i>1 has shape [len(input[1]), len(input[0]), len(input[2]), ..., len(input[i]), ..., 1]
  // Actually all outputs have the same shape in TF meshgrid
  
  // For 2 inputs with shapes [N] and [M]:
  // - output[0] shape: [M, N]
  // - output[1] shape: [M, N]
  
  // Get the length of each input (they should all be 1D for standard meshgrid)
  std::vector<LongType> outputShape(rank);
  for (int i = 0; i < rank; ++i) {
    outputShape[i] = (LongType)shape::length(inputShape->at(i));
  }
  
  // For 'xy' indexing with rank >= 2, swap first two dimensions
  if (swapFirst2Dims && rank >= 2) {
    math::sd_swap<LongType>(outputShape[0], outputShape[1]);
  }
  
  LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  outShapeInfo[0] = rank;
  
  for (int i = 0; i < rank; ++i) {
    outShapeInfo[i + 1] = outputShape[i];
  }

  auto in = inputShape->at(0);
  ShapeUtils::updateStridesAndType(outShapeInfo, in, shape::order(in));

  auto shapes = SHAPELIST();
  auto resultShape = CONSTANT(outShapeInfo);
  
  // All outputs have the same shape
  for (int i = 0; i < rank; ++i) {
    shapes->push_back(resultShape);
  }

  return shapes;
}

}  // namespace ops
}  // namespace sd

#endif
