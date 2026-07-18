/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/DataTypeUtils.h>
#include <helpers/MmulHelper.h>
#include <helpers/shape.h>

#include <vector>

namespace sd {

NDArray* MmulHelper::mmul(NDArray* A, NDArray* B, NDArray* C,
                          const double alpha, const double beta,
                          const char outOrder) {
  LongType lenDim;
  const LongType aRank = A->rankOf();
  const LongType bRank = B->rankOf();
  const bool isAVector = shape::isCommonVector(A->shapeInfo(), lenDim);
  const bool isBVector = shape::isCommonVector(B->shapeInfo(), lenDim);

  if (A->lengthOf() == B->lengthOf() && isAVector && isBVector &&
      (aRank != 2 ||
       (aRank == 2 &&
        (A->isSameShape(B) || (bRank == 1 && A->sizeAt(1) == 1))))) {
    return dot(A, B, C, alpha, beta);
  }

  if (aRank == 2 && bRank == 2) {
    return mmulMxM(A, B, C, alpha, beta, outOrder);
  }

  if (aRank == 2 && isBVector) {
    return mmulMxV(A, B, C, alpha, beta, outOrder);
  }

  if (isAVector && bRank == 2) {
    std::vector<LongType> aShape = {1, A->lengthOf()};
    std::vector<LongType> cShape = {
        1, C == nullptr ? B->sizeAt(1) : C->lengthOf()};

    NDArray* A2 = A->reshape(A->ordering(), aShape);
    NDArray* C2 = C ? C->reshape(C->ordering(), cShape, false) : nullptr;
    auto* result = mmulMxM(A2, B, C2, alpha, beta, outOrder);

    if (A2 != A) delete A2;
    if (C2 != nullptr && C2 != C) delete C2;

    if (C == nullptr) {
      result->reshapei({result->lengthOf()});
      return result;
    }
    return C;
  }

  if ((aRank == 3 || aRank == 4) && aRank == bRank) {
    if (C == nullptr) {
      std::vector<LongType> cShape;
      if (aRank == 3) {
        cShape = {A->sizeAt(0), A->sizeAt(1), B->sizeAt(2)};
      } else {
        cShape = {A->sizeAt(0), A->sizeAt(1), A->sizeAt(2),
                  B->sizeAt(3)};
      }
      C = new NDArray(
          outOrder, cShape,
          DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
          A->getContext());
    }
    if (C->isEmpty()) return C;

    if (mmulBatched(A, B, C, alpha, beta)) {
      return C;
    }
  }

  return mmulNxN(A, B, C, alpha, beta, outOrder);
}

}  // namespace sd
