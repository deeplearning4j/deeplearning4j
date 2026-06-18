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
// Scatter subtract - subtract updates from input at indices
PLATFORM_IMPL(scatter_sub, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto updates = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  // Copy input to output first
  output->assign(input);

  auto outPtr = output->bufferAsT<float>();
  auto updPtr = updates->bufferAsT<float>();

  sd::LongType numIndices = indices->lengthOf();
  sd::LongType sliceSize = input->lengthOf() / input->sizeAt(0);

  for (sd::LongType i = 0; i < numIndices; i++) {
    sd::LongType idx = indices->e<sd::LongType>(i);
    sd::LongType outOffset = idx * sliceSize;
    sd::LongType updOffset = i * sliceSize;

    for (sd::LongType j = 0; j < sliceSize; j++) {
      outPtr[outOffset + j] -= updPtr[updOffset + j];
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(scatter_sub, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE SCATTER_SUB OP");
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
