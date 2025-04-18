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
//  Created by Yurii Shyrma on 25.01.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse_sequence)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(reverse_sequence, 2, 1, false, 0, 2) {
  auto input = INPUT_VARIABLE(0);
  auto seqLengths = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  int seqDim = INT_ARG(0);
  int batchDim = block.numI() > 1 ? INT_ARG(1) : 0;

  REQUIRE_TRUE(input->rankOf() > 1, 0,
               "REVERSE_SEQUENSE operation: input array must have rank > 1, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(seqLengths->rankOf() == 1, 0,
               "REVERSE_SEQUENSE operation: input array seqLengths must be 1D vector, that is it must have rank == 1, "
               "but got %i instead !",
               seqLengths->rankOf());
  REQUIRE_TRUE(seqLengths->lengthOf() == input->sizeAt(batchDim), 0,
               "REVERSE_SEQUENSE custom operation: the length of array seqLengths must be equal to the value of "
               "batchDim dimension of input array, but got %i and %i correspondingly !",
               seqLengths->lengthOf(), input->sizeAt(batchDim));
  REQUIRE_TRUE(seqDim != batchDim, 0,
               "REVERSE_SEQUENSE operation: input integer parameters seqDim and batchDim must be different, but they "
               "both are equal to %i !",
               batchDim);
  REQUIRE_TRUE(batchDim < input->rankOf(), 0,
               "REVERSE_SEQUENSE operation: input integer parameter batchDim must be smaller than input array rank, "
               "but got %i and %i correspondingly !",
               batchDim, input->rankOf());
  REQUIRE_TRUE(seqDim < input->rankOf(), 0,
               "REVERSE_SEQUENSE operation: input integer parameter seqDim must be smaller than input array rank, but "
               "got %i  and %i correspondingly !",
               seqDim, input->rankOf());

  auto maxElem = seqLengths->reduceNumber(reduce::Max);
  REQUIRE_TRUE(maxElem.e<sd::LongType>(0) <= input->sizeAt(seqDim), 0,
               "REVERSE_SEQUENSE operation: max element in seqLengths array must be not greater than value of seqDim "
               "dimension of input array !");

  helpers::reverseSequence(block.launchContext(), input, seqLengths, output, seqDim, batchDim);

  return sd::Status::OK;
}

DECLARE_TYPES(reverse_sequence) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS});
  getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});
  getOpDescriptor()->setAllowedOutputTypes(0, DataType::INHERIT);
}

DECLARE_SHAPE_FN(reverse_sequence) {
  auto inShapeInfo = inputShape->at(0);
  auto seqLenShapeInfo = inputShape->at(1);

  int seqDim = INT_ARG(0);
  int batchDim = block.numI() > 1 ? INT_ARG(1) : 0;

  REQUIRE_TRUE(batchDim < inShapeInfo[0], 0,
               "REVERSE_SEQUENSE operation: input integer parameter batchDim must be smaller than input array rank, "
               "but got %i and %i correspondingly !",
               batchDim, inShapeInfo[0]);
  REQUIRE_TRUE(seqDim < inShapeInfo[0], 0,
               "REVERSE_SEQUENSE operation: input integer parameter seqDim must be smaller than input array rank, but "
               "got %i  and %i correspondingly !",
               seqDim, inShapeInfo[0]);
  REQUIRE_TRUE(inShapeInfo[0] > 1, 0,
               "REVERSE_SEQUENSE operation: input array must have rank > 1, but got %i instead !", inShapeInfo[0]);
  REQUIRE_TRUE(seqLenShapeInfo[0] == 1, 0,
               "REVERSE_SEQUENSE operation: input array seqLengths must be 1D vector, that is it must have rank == 1, "
               "but got %i instead !",
               seqLenShapeInfo[0]);
  REQUIRE_TRUE(seqLenShapeInfo[1] == inShapeInfo[batchDim + 1], 0,
               "REVERSE_SEQUENSE custom operation: the length of array seqLengths must be equal to the value of "
               "batchDim dimension of input array, but got %i and %i correspondingly !",
               seqLenShapeInfo[1], inShapeInfo[batchDim + 1]);

  return SHAPELIST(CONSTANT(inShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
