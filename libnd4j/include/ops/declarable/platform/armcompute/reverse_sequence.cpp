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
// Reverse sequence - reverse elements along seq_dim up to seq_lengths
PLATFORM_IMPL(reverse_sequence, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto seqLengths = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  int seqDim = INT_ARG(0);
  int batchDim = block.numI() > 1 ? INT_ARG(1) : 0;

  // Handle negative dimensions
  if (seqDim < 0) seqDim += input->rankOf();
  if (batchDim < 0) batchDim += input->rankOf();

  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  auto shape = input->shapeOf();
  int rank = input->rankOf();

  std::vector<sd::LongType> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  sd::LongType batchSize = shape[batchDim];
  sd::LongType seqLen = shape[seqDim];

  // Copy input to output first
  std::memcpy(outPtr, inPtr, input->lengthOf() * sizeof(float));

  // For each batch
  for (sd::LongType b = 0; b < batchSize; b++) {
    sd::LongType len = seqLengths->e<sd::LongType>(b);
    if (len <= 1) continue;

    // Calculate number of elements to reverse per batch
    sd::LongType elementsPerSeq = 1;
    for (int d = 0; d < rank; d++) {
      if (d != batchDim && d != seqDim) {
        elementsPerSeq *= shape[d];
      }
    }

    // Reverse along sequence dimension
    for (sd::LongType e = 0; e < elementsPerSeq; e++) {
      for (sd::LongType s = 0; s < len / 2; s++) {
        // Calculate indices for swapping
        sd::LongType idx1 = b * strides[batchDim] + s * strides[seqDim];
        sd::LongType idx2 = b * strides[batchDim] + (len - 1 - s) * strides[seqDim];

        // Add offset for other dimensions
        sd::LongType temp = e;
        for (int d = rank - 1; d >= 0; d--) {
          if (d != batchDim && d != seqDim) {
            sd::LongType coord = temp % shape[d];
            temp /= shape[d];
            idx1 += coord * strides[d];
            idx2 += coord * strides[d];
          }
        }

        // Swap
        float t = outPtr[idx1];
        outPtr[idx1] = outPtr[idx2];
        outPtr[idx2] = t;
      }
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(reverse_sequence, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE REVERSE_SEQUENCE OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
