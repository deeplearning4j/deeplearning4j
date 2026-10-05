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
  // one pass over the indices: an index outside [0, numPartition) belongs to no partition
  const sd::LongType numIndices = indices->lengthOf();
  for (sd::LongType e = 0; e < numIndices; ++e) {
    const sd::LongType partition = indices->e<sd::LongType>(e);
    if (partition >= 0 && partition < numPartition) partitionSizes[partition]++;
  }

  auto shapes = SHAPELIST();
  sd::LongType outRank = shape::rank(in) - shape::rank(idx) + 1;
  for (sd::LongType e = 0; e < numPartition; e++) {
    sd::LongType *newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);
    newShape[0] = outRank;
    newShape[1] = partitionSizes[e];
    // the dimensions of the input beyond the indices' own are the dimensions of a slice
    for (sd::LongType i = 1; i < outRank; ++i) newShape[i + 1] = shape::sizeAt(in, shape::rank(idx) + i - 1);

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

// inputs: the data, the indices and the gradient of every partition (the shape of the partition's output); the
// output is the gradient of the data: every slice of the data takes the slice of its partition's gradient at the
// position of the slice in the partition (the position among the slices of the same partition, in order). Slices
// whose index names no partition were dropped by dynamic_partition and get a zero gradient.
CUSTOM_OP_IMPL(dynamic_partition_bp, 3, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  const auto numPartition = INT_ARG(0);

  auto gradInput = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(numPartition > 0, 0, "dynamic_partition_bp: the number of partitions should be positive, but %i given",
               (int)numPartition);
  REQUIRE_TRUE(block.width() == numPartition + 2, 0,
               "dynamic_partition_bp: the data, the indices and a gradient for each of the %i partitions are expected, "
               "but %i inputs given",
               (int)numPartition, (int)block.width());
  REQUIRE_TRUE(input->rankOf() >= indices->rankOf(), 0,
               "dynamic_partition_bp: data tensor rank should be non-lesser than indices\' tensor, but %i < %i given,",
               input->rankOf(), indices->rankOf());
  for (int dim = 0; dim < indices->rankOf(); dim++) {
    REQUIRE_TRUE(input->sizeAt(dim) == indices->sizeAt(dim), 0,
                 "dynamic_partition_bp: dimensions should be equals for data and indices tensors, but at axis[%i] %i != "
                 "%i given",
                 dim, (int)input->sizeAt(dim), (int)indices->sizeAt(dim));
  }

  // Collect the gradients of the partitions: [slices in the partition, <the dimensions of a slice>]
  const int sliceRank = input->rankOf() - indices->rankOf();
  std::vector<NDArray *> gradOutList(numPartition);
  for (int e = 0; e < numPartition; e++) {
    auto gradient = INPUT_VARIABLE(e + 2);
    gradOutList[e] = gradient;
    REQUIRE_TRUE(gradient->dataType() == input->dataType(), 0,
                 "dynamic_partition_bp: the gradient of partition %i has type %s, but the data has type %s", e,
                 DataTypeUtils::asString(gradient->dataType()).c_str(),
                 DataTypeUtils::asString(input->dataType()).c_str());
    if (gradient->isEmpty()) continue;
    REQUIRE_TRUE(gradient->rankOf() == sliceRank + 1, 0,
                 "dynamic_partition_bp: the gradient of partition %i should have rank %i, but rank %i given", e,
                 sliceRank + 1, gradient->rankOf());
    for (int dim = 0; dim < sliceRank; dim++) {
      REQUIRE_TRUE(gradient->sizeAt(dim + 1) == input->sizeAt(indices->rankOf() + dim), 0,
                   "dynamic_partition_bp: the gradient of partition %i should have the dimensions of a slice of the "
                   "data, but at axis[%i] %i != %i given",
                   e, dim + 1, (int)gradient->sizeAt(dim + 1), (int)input->sizeAt(indices->rankOf() + dim));
    }
  }

  std::vector<NDArray *> outputList = {gradInput};
  helpers::dynamicPartitionFunctorBP(block.launchContext(), input, indices, gradOutList, outputList);

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
