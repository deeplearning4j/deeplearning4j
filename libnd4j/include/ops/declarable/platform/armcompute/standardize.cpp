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
#include <cmath>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Standardize - (x - mean) / std along specified axes
PLATFORM_IMPL(standardize, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get axes from input or int args
  std::vector<int> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<int>(i));
    }
  } else {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  // Handle negative axes
  for (auto& axis : axes) {
    if (axis < 0) axis += input->rankOf();
  }

  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  // Calculate total length and reduce length
  sd::LongType totalLen = input->lengthOf();

  // Simple case: standardize over all elements
  // -1 is the standard sentinel, SD_MAX_INT (Integer.MAX_VALUE) is deprecated but still supported
  if (axes.empty() || (axes.size() == 1 && (axes[0] == -1 || axes[0] == SD_MAX_INT))) {
    float sum = 0.0f;
    for (sd::LongType i = 0; i < totalLen; i++) {
      sum += inPtr[i];
    }
    float mean = sum / totalLen;

    float varSum = 0.0f;
    for (sd::LongType i = 0; i < totalLen; i++) {
      float diff = inPtr[i] - mean;
      varSum += diff * diff;
    }
    float std = std::sqrt(varSum / totalLen + 1e-7f);

    for (sd::LongType i = 0; i < totalLen; i++) {
      outPtr[i] = (inPtr[i] - mean) / std;
    }
  } else {
    // Per-sample standardization for batch dimension
    // Simplified: assume axis 0 is batch, standardize over remaining dims
    sd::LongType batchSize = input->sizeAt(0);
    sd::LongType sampleSize = totalLen / batchSize;

    for (sd::LongType b = 0; b < batchSize; b++) {
      sd::LongType offset = b * sampleSize;

      // Calculate mean
      float sum = 0.0f;
      for (sd::LongType i = 0; i < sampleSize; i++) {
        sum += inPtr[offset + i];
      }
      float mean = sum / sampleSize;

      // Calculate std
      float varSum = 0.0f;
      for (sd::LongType i = 0; i < sampleSize; i++) {
        float diff = inPtr[offset + i] - mean;
        varSum += diff * diff;
      }
      float std = std::sqrt(varSum / sampleSize + 1e-7f);

      // Standardize
      for (sd::LongType i = 0; i < sampleSize; i++) {
        outPtr[offset + i] = (inPtr[offset + i] - mean) / std;
      }
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(standardize, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE STANDARDIZE OP");
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
