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
#include <array/NDArray.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <system/op_boilerplate.h>

#include "../triangular_solve.h"
#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {


/*
 * lower triangular process for system of linear equations
 * x_1 = b_1/a_1,1
 * x_2 = (b_2 - a_2,1 * x_1) / a_2,2
 * x_3 = (b_3 - a_3,1 * x_1 - a_3,2 * x_2) / a_3,3
 * ...
 * x_M = (b_M - a_M,1 * x_1 - ... a_M,M-1 * x_M-1)/ a_M,M
 *
 * output == x
 * a == leftInput
 * b == rightInput
 *
 * */
template <typename T>
static void lowerTriangularSolve(LaunchContext* context, NDArray * leftInput, NDArray * rightInput,
                                 bool const unitsOnDiag, NDArray* output) {

  //TODO: note: this is the cpu implementation.
  //it's not preferred but cuda has enough edge cases
  //that I would prefer to have a working solution for now.

  auto rows = leftInput->rows();
  auto cols = rightInput->columns();
  for (LongType r = 0; r < rows; r++) {
    for (LongType j = 0; j < cols; j++) {
      auto sum = rightInput->t<T>(r, j);

      for (LongType c = 0; c < r; c++) {
        auto left_val = leftInput->t<T>(r, c);
        auto output_val = output->t<T>(c, j);
        sum -= left_val * output_val;

      }



      auto divisor = leftInput->t<T>(r, r);
      output->r<T>(r, j) = unitsOnDiag ? sum : sum / divisor;

    }
  }



}


/*
 * upper triangular process for system of linear equations
 * x_M = b_M/a_M,M
 * x_M-1 = (b_M-1 - a_M-1,M-2 * x_M) / a_M-1,M-1
 * x_M-2 = (b_M-2 - a_M-2,M-3 * x_M-2 - a_M-2,M-1 * x_M) / a_3,3
 * ...
 * x_1 = (b_1 - a_1,2 * x_2 - ... a_1,M * x_M)/ a_1,1
 *
 * output == x
 * a == leftInput
 * b == rightInput
 *
 * */

template <typename T>
static void upperTriangularSolve(LaunchContext* context, NDArray * leftInput, NDArray * rightInput,
                                 bool const unitsOnDiag, NDArray* output) {

  auto rows = leftInput->rows();
  auto cols = rightInput->columns();

  for (LongType r = rows; r > 0; r--) {
    for (LongType j = 0; j < cols; j++) {
      auto sum = rightInput->t<T>(r - 1, j);
      for (LongType c = r; c < rows; c++) {
        sum -= leftInput->t<T>(r - 1, c) * output->t<T>(c, j);
      }

      output->r<T>(r - 1, j) = unitsOnDiag ? sum : sum / leftInput->t<T>(r - 1, r - 1);
    }
  }
}



template <typename T>
static Status triangularSolveFunctor_(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                          bool lower, bool adjoint, NDArray* output) {

  auto leftPart = leftInput->allTensorsAlongDimension({-2, -1});
  auto rightPart = rightInput->allTensorsAlongDimension({-2, -1});
  auto outputPart = output->allTensorsAlongDimension({-2, -1});
  auto batchLoop = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      if(i >= rightPart.size() || i > outputPart.size())
        break;
      if (lower) {
        lowerTriangularSolve<T>(context, leftPart[i], rightPart[i], false, outputPart[i]);
      } else {
        upperTriangularSolve<T>(context, leftPart[i], rightPart[i], false, outputPart[i]);
      }
    }
  };

  samediff::Threads::parallel_tad(batchLoop, 0, leftPart.size(), 1);
  return Status::OK;
}

///  triangularSolve2D - 2D implementation of triangularSolveFunctor
/// \tparam T - type of NDArray output
/// \param context - launch context pointer
/// \param leftInput  - T matrix of equation Tx = b
/// \param rightInput  - b vector of equation Tx = b
/// \param lower - lower or upper triangular matrix
/// \param unitsOnDiag - solve for case when only units (1.0) on diagonal is assumed
/// \param output - output vector (x on equation Tx = b)
///
template <typename T>
void triangularSolve2D(LaunchContext* context, NDArray& leftInput, NDArray& rightInput,
                       bool const lower, bool const unitsOnDiag, NDArray& output) {
  triangularSolveFunctor_<T>(context, const_cast<NDArray*>(&leftInput), const_cast<NDArray*>(&rightInput), lower,
                             unitsOnDiag, &output);


}
BUILD_SINGLE_TEMPLATE(template void triangularSolve2D,
                      (sd::LaunchContext * context, NDArray& leftInput, NDArray& rightInput,
                          bool const lower, bool const unitsOnDiag, NDArray& output),
                      SD_FLOAT_TYPES);

Status triangularSolveFunctor(LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool lower,
                                  bool unitsOnDiag, NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return triangularSolveFunctor_,
                        (context, leftInput, rightInput, lower, unitsOnDiag, output), SD_FLOAT_NATIVE);
}

template <typename T>
static SD_KERNEL void upperAdjointKernel(T const* input, T* output, LongType batchSize, LongType rows, LongType columns,
                                         LongType const* inputTads, LongType const* inputOffsets,
                                         LongType const* outputTads, LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c <= r; c += blockDim.y) {
        LongType zPos[] = {r, c};
        LongType xPos[] = {c, r};
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(inputTads, xPos);
        outputPart[zIndex] = inputPart[xIndex];
      }
    }
  }
}

template <typename T>
static SD_KERNEL void lowerAdjointKernel(T const* input, T* output, LongType batchSize, LongType rows, LongType columns,
                                         LongType const* inputTads, LongType const* inputOffsets,
                                         LongType const* outputTads, LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = r + threadIdx.y; c < columns; c += blockDim.y) {
        LongType zPos[] = {r, c};
        LongType xPos[] = {c, r};
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(inputTads, xPos);
        outputPart[zIndex] = inputPart[xIndex];
      }
    }
  }
}

template <typename T>
static void adjointTriangularMatrix_(LaunchContext* context, NDArray * input, bool const lower,
                                     NDArray* output) {
  NDArray::prepareSpecialUse({input}, {output});
  std::vector<LongType> dims = {-2, -1};
  auto inputTads = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims);
  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(),&dims);
  auto stream = context->getCudaStream();
  auto inputBuf = reinterpret_cast<T const*>(input->specialBuffer());
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);
  dim3 launchDims = getLaunchDims("triangular_solve");
  if (lower) {
    lowerAdjointKernel<T><<<launchDims.y, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                                 inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                                                 outputTads->specialShapeInfo(), outputTads->specialOffsets());
    sd::DebugHelper::checkErrorCode(stream, "lowerAdjointKernel failed");

  } else {
    upperAdjointKernel<T><<<launchDims.y, launchDims.x,launchDims.z, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                                inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                                                outputTads->specialShapeInfo(), outputTads->specialOffsets());
    sd::DebugHelper::checkErrorCode(stream, "upperAdjointKernel failed");

  }

  NDArray::registerSpecialUse({input}, {output});
}

void adjointMatrix(LaunchContext* context, NDArray * input, bool const lower, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointTriangularMatrix_, (context, input, lower, output), SD_FLOAT_NATIVE);
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd
