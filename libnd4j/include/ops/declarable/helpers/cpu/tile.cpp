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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <helpers/Loops.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_tile)
namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void tileBP_(NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<sd::LongType> reps) {
  T* gradIBuff = reinterpret_cast<T*>(gradI.buffer());
  auto gradOBuff = reinterpret_cast<T const*>(gradO.buffer());
  const sd::LongType gradILen = gradI.lengthOf();
  const sd::LongType gradOLen = gradO.lengthOf();  // gradOLen >= gradILen

  // initial zeroing of gradI content
  sd::ops::safe_zero(gradIBuff, static_cast<size_t>(gradILen));

  LongType gradOCoords[SD_MAX_RANK];
  LongType gradICoords[SD_MAX_RANK];
  LongType gradOOffset;
  LongType gradIOffset;

  sd::LongType gradORank = gradO.rankOf();
  sd::LongType *gradOShape = shape::shapeOf(gradO.shapeInfo());
  sd::LongType *gradOStride = shape::stride(gradO.shapeInfo());
  sd::LongType gradIRank = gradI.rankOf();
  sd::LongType *gradIShape = shape::shapeOf(gradI.shapeInfo());
  sd::LongType *gradIStride = shape::stride(gradI.shapeInfo());
  for (sd::LongType i = 0; i < gradOLen; ++i) {
    INDEX2COORDS(i,gradORank, gradOShape, gradOCoords);
    COORDS2INDEX(gradORank, gradOStride, gradOCoords, gradOOffset);
    INDEX2COORDS(i, gradIRank, gradIShape, gradICoords);
    COORDS2INDEX(gradIRank, gradIStride, gradICoords, gradIOffset);
    gradI.p(gradIOffset, gradI.e<T>(gradIOffset) + gradOBuff[gradOOffset]);
  }
}

void tileBP(LaunchContext* context, NDArray gradO /*input*/, NDArray& gradI /*output*/,
            const std::vector<LongType> reps) {
  BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (gradO, gradI, reps), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void tileBP_,
                      (NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<sd::LongType> reps),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif