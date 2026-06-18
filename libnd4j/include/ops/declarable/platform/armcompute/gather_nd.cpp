/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// GatherNd - gather slices from input using indices
PLATFORM_IMPL(gather_nd, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  auto inputShape = input->shapeOf();
  auto indicesShape = indices->shapeOf();
  int inputRank = input->rankOf();
  int indicesRank = indices->rankOf();

  // Last dimension of indices determines how many dimensions we index into
  sd::LongType indexDepth = indicesShape[indicesRank - 1];

  // Calculate input strides
  std::vector<sd::LongType> inputStrides(inputRank);
  inputStrides[inputRank - 1] = 1;
  for (int i = inputRank - 2; i >= 0; i--) {
    inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
  }

  // Size of each slice (elements after the indexed dimensions)
  sd::LongType sliceSize = 1;
  for (int i = indexDepth; i < inputRank; i++) {
    sliceSize *= inputShape[i];
  }

  // Number of indices (all dimensions except the last one)
  sd::LongType numIndices = 1;
  for (int i = 0; i < indicesRank - 1; i++) {
    numIndices *= indicesShape[i];
  }

  // For each index tuple
  for (sd::LongType idx = 0; idx < numIndices; idx++) {
    // Calculate the input offset for this index tuple
    sd::LongType inputOffset = 0;
    for (sd::LongType d = 0; d < indexDepth; d++) {
      sd::LongType indexVal = indices->e<sd::LongType>(idx * indexDepth + d);
      inputOffset += indexVal * inputStrides[d];
    }

    // Copy the slice
    sd::LongType outputOffset = idx * sliceSize;
    for (sd::LongType s = 0; s < sliceSize; s++) {
      outPtr[outputOffset + s] = inPtr[inputOffset + s];
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(gather_nd, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE GATHER_ND OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
