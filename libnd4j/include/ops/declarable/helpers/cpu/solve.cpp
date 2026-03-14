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

//
//  @author GS <sgazeos@gmail.com>
//
#include "../solve.h"

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <system/op_boilerplate.h>

#include "../lup.h"
#include "../triangular_solve.h"
#if NOT_EXCLUDED(OP_solve)
namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------------------------------------------------------------------
// //
template <typename T>
static void adjointMatrix_(sd::LaunchContext* context, NDArray * input, NDArray* output) {
  auto rows = input->sizeAt(-2);
  output->assign(input);

  // For unbatched (2D) inputs, allTensorsAlongDimension({-2,-1}) produces rank-0 TADs
  // which breaks coordinate-based indexing. Use the arrays directly instead.
  if (input->rankOf() == 2) {
    for (sd::LongType r = 0; r < rows; r++) {
      for (sd::LongType c = 0; c < r; c++) {
        math::sd_swap(output->r<T>(r, c), output->r<T>(c, r));
      }
    }
    return;
  }

  auto inputPart = input->allTensorsAlongDimension({-2, -1});
  auto outputPart = output->allTensorsAlongDimension({-2, -1});

  auto batchLoop = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; batch++) {
      for (sd::LongType r = 0; r < rows; r++) {
        for (sd::LongType c = 0; c < r; c++) {
          math::sd_swap(outputPart[batch]->r<T>(r, c), outputPart[batch]->r<T>(c, r));
        }
      }
    }
  };
  samediff::Threads::parallel_tad(batchLoop, 0, inputPart.size(), 1);
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// //
template <typename T>
static sd::Status solveFunctor_(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool const adjoint,
                                NDArray* output) {
  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();
  auto permuShape = rightInput->getShapeAsVector();
  permuShape->pop_back();

  // For non-batched (2D) inputs, pop_back() leaves a 1D shape (e.g., [3]).
  // allTensorsAlongDimension({-1}) on a 1D array doesn't properly decompose
  // into individual vectors. Ensure the shape is at least 2D by prepending a 1
  // so that the batch decomposition works correctly (e.g., [3] -> [1, 3]).
  if (permuShape->size() == 1) {
    permuShape->insert(permuShape->begin(), 1);
  }

  std::vector<sd::LongType> &shapeDeRef = *permuShape;
  auto permutations = NDArrayFactory::create<sd::LongType>('c', shapeDeRef, context);
  helpers::lu(context, leftInput, leftOutput, permutations);


  auto P = leftInput->ulike();  // permutations batched matrix
  P->nullify();                  // to fill up matrices with zeros

  // For unbatched (2D) inputs, allTensorsAlongDimension({-2,-1}) produces rank-0 TADs
  // which breaks coordinate-based indexing. Use the arrays directly instead.
  bool unbatched = (leftInput->rankOf() == 2);
  if (unbatched) {
    for (sd::LongType row = 0; row < P->rows(); row++) {
      P->r<T>(row, permutations->t<sd::LongType>(row)) = T(1.f);
    }
  } else {
    auto PPart = P->allTensorsAlongDimension({-2, -1});
    auto permutationsPart = permutations->allTensorsAlongDimension({-1});
    for (auto batch = 0; batch < permutationsPart.size(); batch++) {
      for (sd::LongType row = 0; row < PPart[batch]->rows(); row++) {
        PPart[batch]->r<T>(row, permutationsPart[batch]->t<sd::LongType>(row)) = T(1.f);
      }
    }
  }

  auto leftLower = leftOutput->dup(leftOutput->ordering());
  auto rightOutput = rightInput->ulike();
  auto rightPart = rightInput->ulike();

  MmulHelper::matmul(P, rightInput, rightPart, false, false, 1.0, 0.0, rightPart);

  // Set diagonal of lower triangular part to 1
  if (unbatched) {
    for (sd::LongType r = 0; r < leftLower->rows(); r++) leftLower->r<T>(r, r) = (T)1.f;
  } else {
    ResultSet leftLowerPart = leftLower->allTensorsAlongDimension({-2, -1});
    for (auto i = 0; i < leftLowerPart.size(); i++) {
      for (sd::LongType r = 0; r < leftLowerPart[i]->rows(); r++) leftLowerPart[i]->r<T>(r, r) = (T)1.f;
    }
  }

  // stage 2: triangularSolveFunctor for Lower with given b
  helpers::triangularSolveFunctor(context, leftLower, rightPart, true, false, rightOutput);
  // stage 3: triangularSolveFunctor for Upper with output of previous stage
  helpers::triangularSolveFunctor(context, leftOutput, rightOutput, false, false, output);

  delete permutations;
  delete permuShape;
  return sd::Status::OK;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// //
sd::Status solveFunctor(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool const adjoint,
                        NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return solveFunctor_, (context, leftInput, rightInput, adjoint, output),
                        SD_FLOAT_TYPES);
}
// ---------------------------------------------------------------------------------------------------------------------------------------
// //
void adjointMatrix(sd::LaunchContext* context, NDArray * input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
}
// ---------------------------------------------------------------------------------------------------------------------------------------
// //
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif