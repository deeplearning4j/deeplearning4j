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
// @author Yurii Shyrma, created on 21.09.2018
// @author raver119@gmail.com
//

#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>

#include <ops/declarable/helpers/ismax.h>
#if NOT_EXCLUDED(OP_ismax)
namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Z>
static void ismax_(NDArray* input, NDArray* output, const std::vector<LongType>& dimensions) {
  if (input->isVector()) {
    int dimensionsLength = dimensions.size();
    int length = input->lengthOf();
    if (!dimensions.empty() && (input->shapeOf())[dimensions[0]] == 1) {
      for (int i = 0; i < length; i++) output->p<Z>(i, static_cast<Z>(1));
    } else {
      int maxIdx = 0;
      auto currMax = input->e<X>(0);
      if (length < ELEMENT_THRESHOLD) {
        for (int i = 0; i < length; i++) {
          if (currMax < input->e<X>(i)) {
            currMax = input->e<X>(i);
            maxIdx = i;
          }
          output->p<Z>(i, static_cast<Z>(0));
        }
      } else {
        {
          int maxIdxLocal = maxIdx;
          auto currMaxLocal = currMax;

          for (int i = 0; i < length; i++) {
            if (currMaxLocal < input->e<X>(i)) {
              currMaxLocal = input->e<X>(i);
              maxIdxLocal = i;
            }
            output->p<Z>(i, static_cast<Z>(0));
          }

          PRAGMA_OMP_CRITICAL {
            if (currMax < currMaxLocal) {
              currMax = currMaxLocal;
              maxIdx = maxIdxLocal;
            }
          }
        }
      }
      output->p<Z>(maxIdx, static_cast<Z>(1));
    }
  } else {
    int dimensionsLength = dimensions.size();
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(
        input->shapeInfo(), const_cast<sd::LongType *>(dimensions.data()), dimensionsLength);
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(
        output->shapeInfo(), const_cast<sd::LongType *>(dimensions.data()), dimensionsLength);

    auto tadShapeShapeInfo = tadPack->primaryShapeInfo();
    auto tadOffsets = tadPack->primaryOffsets();
    auto zOfsets = tadPackZ->platformOffsets();

    int tadLength = shape::length(tadShapeShapeInfo);
    int tads = tadPack->numberOfTads();

    int tadsPerThread = tads / TAD_THRESHOLD;
    int num_threads = sd::math::sd_max<int>(1, tadsPerThread);
    num_threads = sd::math::sd_min<int>(num_threads, omp_get_max_threads());

    int span = (tads / num_threads) + 8;

    auto func = PRAGMA_THREADS_FOR {
      for (auto r = start; r < stop; r++) {
        auto rX = const_cast<NDArray*>(input)->bufferAsT<X>() + tadOffsets[r];
        auto rZ = output->bufferAsT<Z>() + zOfsets[r];

        auto maxValue = rX[0];
        int maxIdx = 0;
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xOffset;
        LongType zOffset;
        sd::LongType tadRank = shape::rank(tadShapeShapeInfo);
        sd::LongType *tadShape = shape::shapeOf(tadShapeShapeInfo);
        sd::LongType *tadStride = shape::stride(tadShapeShapeInfo);
        for (sd::LongType i = 0; i < tadLength; i++) {
          INDEX2COORDS(i,tadRank,tadShape, xCoords);
          COORDS2INDEX(tadRank, tadStride, xCoords, xOffset);
          if (rX[xOffset] > maxValue) {
            maxIdx = i;
            maxValue = rX[xOffset];
          }
        }

        sd::LongType tadPackZRank = shape::rank(tadPackZ->primaryShapeInfo());
        sd::LongType *tadPackZShape = shape::shapeOf(tadPackZ->primaryShapeInfo());
        sd::LongType *tadPackZStride = shape::stride(tadPackZ->primaryShapeInfo());
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < tadLength; i++) {
          INDEX2COORDS(i, tadPackZRank, tadPackZShape, zCoords);
          COORDS2INDEX(tadPackZRank, tadPackZStride, zCoords, zOffset);
          rZ[zOffset] = maxIdx == i ? (Z)1 : (Z)0;
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, tads);
  }
}

void ismax(sd::LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>& dimensions) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), ismax_, (input, output, dimensions), SD_COMMON_TYPES,
                        SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif