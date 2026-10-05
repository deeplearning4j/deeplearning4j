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

#include <execution/Threads.h>
#include <helpers/Loops.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/op_types.h>
#if NOT_EXCLUDED(OP_tile)
namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// The gradient of an entry of the tiled array's input is the sum of the entries of gradO it was tiled into: the entries
// at its own coordinates plus a whole number of shapes of gradI in each dimension. Each entry of gradI is summed alone,
// in AggregateType<T>, through the strides of both arrays, so any layout of either is read and written as it is.
template <typename T>
static void tileBP_(NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<sd::LongType> reps) {
  (void)reps;
  using AccT = typename simdOps::AggregateType<T>::type;

  const sd::LongType gradILen = gradI.lengthOf();
  if (gradILen == 0) return;

  const T* gradOBuff = gradO.bufferAsT<T>();
  T* gradIBuff = gradI.bufferAsT<T>();

  const sd::LongType rank = gradI.rankOf();  // gradO has the same rank
  const sd::LongType* gradIShape = shape::shapeOf(gradI.shapeInfo());
  const sd::LongType* gradIStride = shape::stride(gradI.shapeInfo());
  const sd::LongType* gradOShape = shape::shapeOf(gradO.shapeInfo());
  const sd::LongType* gradOStride = shape::stride(gradO.shapeInfo());

  // how many times gradI's shape is repeated along each dimension of gradO
  sd::LongType repeats[SD_MAX_RANK];
  sd::LongType numRepeats = 1;
  for (sd::LongType d = 0; d < rank; ++d) {
    repeats[d] = gradOShape[d] / gradIShape[d];
    numRepeats *= repeats[d];
  }

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType repeat[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, rank, gradIShape, coords);

      sd::LongType gradIOffset;
      sd::LongType gradOOffset;
      COORDS2INDEX(rank, gradIStride, coords, gradIOffset);
      // the first of the entries of gradO: the same coordinates
      COORDS2INDEX(rank, gradOStride, coords, gradOOffset);

      for (sd::LongType d = 0; d < rank; ++d) repeat[d] = 0;

      AccT sum = static_cast<AccT>(0);
      for (sd::LongType n = 0; n < numRepeats; ++n) {
        sum += static_cast<AccT>(gradOBuff[gradOOffset]);

        // on to the next repeat: the last dimension first
        for (sd::LongType d = rank - 1; d >= 0; --d) {
          if (++repeat[d] < repeats[d]) {
            gradOOffset += gradIShape[d] * gradOStride[d];
            break;
          }
          gradOOffset -= (repeats[d] - 1) * gradIShape[d] * gradOStride[d];
          repeat[d] = 0;
        }
      }

      gradIBuff[gradIOffset] = static_cast<T>(sum);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradILen);
}

void tileBP(LaunchContext* context, NDArray gradO /*input*/, NDArray& gradI /*output*/,
            const std::vector<LongType> reps) {
  BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (gradO, gradI, reps), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void tileBP_,
                      (NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<sd::LongType> reps),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
