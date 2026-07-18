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
//  @author GS <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dynamic_partition)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/dynamic.h>

#include <array>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(dynamic_partition, 2, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);

  REQUIRE_TRUE(input->rankOf() >= indices->rankOf(), 0,
               "dynamic_partition: data tensor rank should be non-lesser than indices\' tensor, but %i < %i given,",
               input->rankOf(), indices->rankOf());
  for (int dim = 0; dim < indices->rankOf(); dim++) {
    REQUIRE_TRUE(
        input->sizeAt(dim) == indices->sizeAt(dim), 0,
        "dynamic_partition: dimensions should be equals for data and indices tensors, but at axis[%i] %i != %i given",
        dim, input->sizeAt(dim), indices->sizeAt(dim));
  }

  auto numPartition = INT_ARG(0);
  std::vector<NDArray *> outputList(numPartition);
  for (int o = 0; o < numPartition; ++o) {
    outputList[o] = OUTPUT_VARIABLE(o);
  }
  helpers::dynamicPartitionFunctor(block.launchContext(), input, indices, outputList);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dynamic_partition) {
  auto numPartition = INT_ARG(0);
  auto indices = INPUT_VARIABLE(1);
  std::vector<sd::LongType> partitionSizes(numPartition, 0);
  auto in = inputShape->at(0);
  auto idx = inputShape->at(1);
  for (int i = 0; i < numPartition; i++) {
    for (int e = 0; e < indices->lengthOf(); ++e)
      if (indices->e<sd::LongType>(e) == i) partitionSizes[i]++;
  }

  auto shapes = SHAPELIST();
  sd::LongType outRank = shape::rank(in) - shape::rank(idx) + 1;
  for (sd::LongType e = 0; e < numPartition; e++) {
    sd::LongType *newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);
    newShape[0] = outRank;
    newShape[1] = partitionSizes[e];
    for (sd::LongType i = 1; i < outRank; ++i) newShape[i + 1] = shape::sizeAt(in, outRank + i - 1);

    shape::updateStrides(newShape, shape::order(in), false);
    ArrayOptions::setDataType(newShape, ArrayOptions::dataType(in));
    shapes->push_back(CONSTANT(newShape));
  }

  return shapes;
}

DECLARE_TYPES(dynamic_partition) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_SPLIT | OP_TRAIT_DATA_DEPENDENT);
}

DECLARE_TYPES(dynamic_partition_bp) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true);  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_SPLIT | OP_TRAIT_BACKWARD | OP_TRAIT_DATA_DEPENDENT); }

CUSTOM_OP_IMPL(dynamic_partition_bp, 3, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto numPartition = INT_ARG(0);

  auto gradInput = OUTPUT_VARIABLE(0);

  // Collect gradients from each partition output
  std::vector<NDArray *> gradOutList(numPartition);
  for (sd::LongType e = 0; e < numPartition; e++) {
    gradOutList[e] = INPUT_VARIABLE(e + 2);
  }

  // Track position within each partition
  std::vector<sd::LongType> partitionCounters(numPartition, 0);

  // Scatter gradients back to original positions
  // For each element i: grad_input[i] = grad_partition[partition[i]][position_within_partition[i]]
  auto len = indices->lengthOf();
  for (sd::LongType i = 0; i < len; i++) {
    auto partitionIdx = indices->e<sd::LongType>(i);
    REQUIRE_TRUE(partitionIdx >= 0 && partitionIdx < numPartition, 0,
                 "dynamic_partition_bp: partition index %lld out of range [0, %d)", partitionIdx, numPartition);
    auto posInPartition = partitionCounters[partitionIdx]++;
    auto gradVal = gradOutList[partitionIdx]->e<double>(posInPartition);
    gradInput->p(i, gradVal);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dynamic_partition_bp) {
  auto shapes = SHAPELIST();

  auto inputShapeInfo = inputShape->at(0);
  shapes->push_back(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inputShapeInfo), shape::order(inputShapeInfo),
      shape::rank(inputShapeInfo), shape::shapeOf(inputShapeInfo), 0));

  return shapes;
}
}  // namespace ops
}  // namespace sd

#endif
