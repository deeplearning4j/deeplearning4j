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
// Mirror Pad - pad with mirrored values
PLATFORM_IMPL(mirror_pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);  // [rank, 2] - before and after padding for each dim
  auto output = OUTPUT_VARIABLE(0);

  int mode = INT_ARG(0);  // 0 = REFLECT, 1 = SYMMETRIC

  auto inShape = input->shapeOf();
  auto outShape = output->shapeOf();
  int rank = input->rankOf();

  // Get padding values
  std::vector<sd::LongType> padBefore(rank), padAfter(rank);
  for (int d = 0; d < rank; d++) {
    padBefore[d] = paddings->e<sd::LongType>(d, 0);
    padAfter[d] = paddings->e<sd::LongType>(d, 1);
  }

  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  // Calculate strides
  std::vector<sd::LongType> inStrides(rank), outStrides(rank);
  inStrides[rank - 1] = 1;
  outStrides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; d--) {
    inStrides[d] = inStrides[d + 1] * inShape[d + 1];
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];
  }

  // For each output element
  for (sd::LongType outLinear = 0; outLinear < output->lengthOf(); outLinear++) {
    // Convert linear index to coordinates
    std::vector<sd::LongType> outCoord(rank);
    sd::LongType temp = outLinear;
    for (int d = rank - 1; d >= 0; d--) {
      outCoord[d] = temp % outShape[d];
      temp /= outShape[d];
    }

    // Map to input coordinates with mirroring
    std::vector<sd::LongType> inCoord(rank);
    for (int d = 0; d < rank; d++) {
      sd::LongType coord = outCoord[d] - padBefore[d];

      if (coord < 0) {
        // Mirror before
        coord = mode == 0 ? -coord : -coord - 1;
      } else if (coord >= inShape[d]) {
        // Mirror after
        sd::LongType excess = coord - inShape[d];
        coord = mode == 0 ? inShape[d] - 2 - excess : inShape[d] - 1 - excess;
      }

      // Clamp to valid range
      coord = std::max(0LL, std::min(coord, inShape[d] - 1));
      inCoord[d] = coord;
    }

    // Calculate input linear index
    sd::LongType inLinear = 0;
    for (int d = 0; d < rank; d++) {
      inLinear += inCoord[d] * inStrides[d];
    }

    outPtr[outLinear] = inPtr[inLinear];
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(mirror_pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE MIRROR_PAD OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
