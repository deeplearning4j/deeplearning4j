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
//  @author sgazeos@gmail.com
//
#include <ops/declarable/helpers/axis.h>

namespace sd {
namespace ops {
namespace helpers {

void adjustAxis(sd::LongType rank, NDArray* axisVector, std::vector<LongType>& output) {
  // Guard: axisVector may be null when called from DECLARE_SHAPE_FN during the
  // DSP shape pre-pass (phaseShapeInferenceOnly), where _fastpath_in is null-padded
  // to slot.wiring.numInputs so that block.width() reflects the full input count.
  // INPUT_VARIABLE(2) returns _fastpath_in[2] which can be nullptr for an absent
  // optional axes input.  Dereferencing null here causes SIGSEGV (NDArray::shapeInfo
  // with this=0x0, si_addr=0x10).  When axisVector is absent, leave `output` empty;
  // the caller (DECLARE_SHAPE_FN) already falls back to iArgs for the axis.
  if (axisVector == nullptr) return;
  if(axisVector->isScalar()) {
    output.resize(1);
    auto ca = axisVector->e<sd::LongType>(0);
    if (ca < 0)  // shift values on rank for negative vals
      ca += rank;
    output[0] = ca;
    return;
  }
  output.resize(axisVector->lengthOf());
  axisVector->tickReadDevice();  // mark input as read on device
  for (int e = 0; e < axisVector->lengthOf(); e++) {
    auto ca = axisVector->e<sd::LongType>(e);
    if (ca < 0)  // shift values on rank for negative vals
      ca += rank;

    output[e] = ca;
  }
}

void adjustAxis(sd::LongType rank, std::vector<LongType>& axisVector) {
  for (size_t e = 0; e < axisVector.size(); e++) {
    auto a = axisVector[e];
    if (a < 0) axisVector[e] = a + rank;
  }
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
