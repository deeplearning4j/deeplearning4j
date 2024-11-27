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
//  @author raver119@gmail.com
//
#include <ops/declarable/helpers/flatten.h>
#if NOT_EXCLUDED(OP_flatten)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void flatten_(std::vector<NDArray *> &inputs, NDArray *output, const char order) {
  int numArrays = inputs.size();
  std::vector<sd::LongType> offsets(numArrays);
  sd::LongType cOffset = 0;

  // calculating offsets in output
  for (int e = 0; e < numArrays; e++) {
    offsets[e] = cOffset;
    cOffset += inputs[e]->lengthOf();
  }

  // actually transferring data
  for (sd::LongType e = 0; e < numArrays; e++) {
    auto z = reinterpret_cast<T *>(output->bufferWithOffset(offsets[e]));

    auto xBuffer = inputs[e]->bufferAsT<T>();
    auto xShapeInfo = inputs[e]->shapeInfo();
    auto xLength = inputs[e]->lengthOf();

    for (sd::LongType i = 0; i < xLength; i++) {
      sd::LongType xOffset;
      sd::LongType xCoords[SD_MAX_RANK];
      INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, xCoords);
      COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), xCoords, xOffset);
      z[i] = xBuffer[xOffset];
    }
  }
}
void flatten(sd::LaunchContext *context, std::vector<NDArray *> &inputs, NDArray *output, char order) {
  BUILD_SINGLE_SELECTOR(output->dataType(), flatten_, (inputs, output, order), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif