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
#if NOT_EXCLUDED(OP_scatter_update)
namespace sd {
namespace ops {
namespace helpers {

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
      if (inSubArr.lengthOf() != updSubArr.lengthOf()) continue;

      switch (opCode) {
        case 0:
          inSubArr.applyPairwiseTransform(pairwise::Add, &updSubArr, &inSubArr);
          break;
        case 1:
          inSubArr.applyPairwiseTransform(pairwise::Subtract, &updSubArr, &inSubArr);
          break;
        case 2:
          inSubArr.applyPairwiseTransform(pairwise::Multiply, &updSubArr, &inSubArr);
          break;
        case 3:
          inSubArr.applyPairwiseTransform(pairwise::Divide, &updSubArr, &inSubArr);
          break;
        case 4:
          inSubArr.applyPairwiseTransform(pairwise::ReverseSubtract, &updSubArr, &inSubArr);
          break;
        case 5:
          inSubArr.applyPairwiseTransform(pairwise::ReverseDivide, &updSubArr, &inSubArr);
          break;
        case 6:
          inSubArr.applyPairwiseTransform(pairwise::CopyPws, &updSubArr, &inSubArr);
          break;
        default:
          continue;
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, indices.size());


  delete dimsToExclude;
}

//////////////////////////////////////////////////////////////////////////
void scatterSimple(sd::LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                   NDArray& indices, const std::vector<LongType>& dimensions) {
  // updates and indices have same length
  const sd::LongType len = indices.lengthOf();

  switch (opId) {
    case 6: {  // copy
      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          auto inSubArr = input(i, dimensions);
          auto curr = indices.e(i);
          inSubArr.p(indices.t<sd::LongType>(i), &curr);
        }
      };

      samediff::Threads::parallel_for(func, 0, len);
    } break;

    default:
      THROW_EXCEPTION("helpers::scatterSimple: operation is not implemented for given id !");
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif