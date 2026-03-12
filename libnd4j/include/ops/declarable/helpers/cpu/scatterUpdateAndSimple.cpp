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
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_scatter_update)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void scatterSimpleCopy(NDArray& input, NDArray& updates, NDArray& indices,
                              const std::vector<LongType>& tadDimensions) {
  if (tadDimensions.empty()) {
    THROW_EXCEPTION("helpers::scatterSimple: TAD dimensions must not be empty for copy operation");
  }

  auto tadPack =
      ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), const_cast<LongType*>(tadDimensions.data()),
                                                        tadDimensions.size());
  auto* tadShapeInfo = tadPack->primaryShapeInfo();
  auto* tadOffsets = tadPack->primaryOffsets();
  const auto numTads = tadPack->numberOfTads();
  const auto tadLen = shape::length(tadShapeInfo);
  const auto tadRank = shape::rank(tadShapeInfo);
  const auto* tadShape = shape::shapeOf(tadShapeInfo);
  const auto* tadStride = shape::stride(tadShapeInfo);

  if (indices.lengthOf() != updates.lengthOf()) {
    THROW_EXCEPTION("helpers::scatterSimple: indices and updates must have the same length");
  }
  if (indices.lengthOf() != numTads) {
    THROW_EXCEPTION("helpers::scatterSimple: number of indices/updates must match number of TADs");
  }

  auto* inputBuffer = input.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    for (auto t = start; t < stop; t++) {
      const auto localIndex = indices.t<sd::LongType>(t);
      if (localIndex < 0 || localIndex >= tadLen) {
        THROW_EXCEPTION("helpers::scatterSimple: local index is out of bounds for TAD length");
      }

      sd::LongType coords[SD_MAX_RANK];
      sd::LongType localOffset = 0;
      INDEX2COORDS(localIndex, tadRank, tadShape, coords);
      COORDS2INDEX(tadRank, tadStride, coords, localOffset);

      inputBuffer[tadOffsets[t] + localOffset] = updates.e<T>(t);
    }
  };

  samediff::Threads::parallel_for(func, 0, numTads);
}

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(sd::LaunchContext* context, NDArray& input, NDArray& updates, const std::vector<LongType>* intArgs) {
  sd::LongType opCode = (*intArgs)[0];
  sd::LongType dimSize = (*intArgs)[1];
  sd::LongType e;
  sd::LongType limg = 2 + dimSize;
  std::vector<sd::LongType> tadDimensions(dimSize);
  for (e = 2; e < limg; e++) tadDimensions[e - 2] = (*intArgs)[e];

  std::vector<sd::LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(), tadDimensions.size(),tadDimensions.data());

  // increasing counter to skip numIndices
  e++;
  std::vector<sd::LongType> indices;
  for (; e < static_cast<sd::LongType>(intArgs->size()); e++) indices.push_back((*intArgs)[e]);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto inSubArr = input(indices[i], *dimsToExclude, true);
      auto updSubArr = updates(i, *dimsToExclude, true);
      if (inSubArr->lengthOf() != updSubArr->lengthOf())  {
        delete inSubArr;
        continue;
      }

      switch (opCode) {
        case 0:
          inSubArr->applyPairwiseTransform(pairwise::Add, updSubArr, inSubArr);
          break;
        case 1:
          inSubArr->applyPairwiseTransform(pairwise::Subtract, updSubArr, inSubArr);
          break;
        case 2:
          inSubArr->applyPairwiseTransform(pairwise::Multiply, updSubArr, inSubArr);
          break;
        case 3:
          inSubArr->applyPairwiseTransform(pairwise::Divide, updSubArr, inSubArr);
          break;
        case 4:
          inSubArr->applyPairwiseTransform(pairwise::ReverseSubtract, updSubArr, inSubArr);
          break;
        case 5:
          inSubArr->applyPairwiseTransform(pairwise::ReverseDivide, updSubArr, inSubArr);
          break;
        case 6:
          inSubArr->applyPairwiseTransform(pairwise::CopyPws, updSubArr, inSubArr);
          break;
        default:
          continue;
      }

      delete inSubArr;
      delete updSubArr;
    }
  };

  samediff::Threads::parallel_tad(func, 0, indices.size());


  delete dimsToExclude;
}

//////////////////////////////////////////////////////////////////////////
void scatterSimple(sd::LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                   NDArray& indices, const std::vector<LongType>& dimensions) {
  switch (opId) {
    case 6: {  // copy
      BUILD_SINGLE_SELECTOR(input.dataType(), scatterSimpleCopy, (input, updates, indices, dimensions), SD_COMMON_TYPES);
    } break;

    default:
      THROW_EXCEPTION("helpers::scatterSimple: operation is not implemented for given id !");
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
