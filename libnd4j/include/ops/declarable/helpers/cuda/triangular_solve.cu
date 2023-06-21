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
static SD_HOST_DEVICE void lowerTriangularSolve(T const* leftInput, sd::LongType const* leftInputShape,
                                                T const* rightInput, sd::LongType const* rightInputShape,
                                                bool const unitOnDiag, T* output, const sd::LongType* outputShape,
                                                sd::LongType rows, sd::LongType cols) {
  for (auto r = 0; r < rows; r++) {
    for (auto j = 0; j < cols; j++) {
      sd::LongType posY[] = {r, j};
      sd::LongType posX[] = {r, r};
      auto xIndex = shape::getOffset(leftInputShape, posX, 0);
      auto yIndex = shape::getOffset(rightInputShape, posY, 0);
      auto zIndex = shape::getOffset(outputShape, posY, 0);

      auto sum = rightInput[yIndex];
      for (auto c = 0; c < r; c++) {
        sd::LongType posZ[] = {c, j};
        sd::LongType pos[] = {r, c};
        auto xcIndex = shape::getOffset(leftInputShape, pos, 0);
        auto zcIndex = shape::getOffset(outputShape, posZ, 0);
        sum -= leftInput[xcIndex] * output[zcIndex];
      }
      output[zIndex] = unitOnDiag ? sum : sum / leftInput[xIndex];
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
static SD_HOST_DEVICE void upperTriangularSolve(T const* leftInput, sd::LongType const* leftInputShape,
                                                T const* rightInput, sd::LongType const* rightInputShape,
                                                bool const unitOnDiag, T* output, const sd::LongType* outputShape,
                                                sd::LongType rows, sd::LongType cols) {
  for (auto r = rows; r > 0; r--) {
    for (auto j = 0; j < cols; j++) {
      sd::LongType posY[] = {r - 1, j};
      sd::LongType posX[] = {r - 1, r - 1};
      auto xIndex = shape::getOffset(leftInputShape, posX, 0);
      auto yIndex = shape::getOffset(rightInputShape, posY, 0);
      auto zIndex = shape::getOffset(outputShape, posY, 0);
      auto sum = rightInput[yIndex];
      for (auto c = r; c < rows; c++) {
        sd::LongType posZ[] = {c, j};
        sd::LongType pos[] = {r - 1, c};
        auto zcIndex = shape::getOffset(outputShape, posZ, 0);
        auto xcIndex = shape::getOffset(leftInputShape, pos, 0);
        sum -= leftInput[xcIndex] * output[zcIndex];
      }
      output[zIndex] = unitOnDiag ? sum : sum / leftInput[xIndex];
    }
  }
}

template <typename T>
static SD_KERNEL void triangularSolveKernel(T const* leftInput, sd::LongType const* leftPartShape, T const* rightInput,
                                            sd::LongType const* rightPartShape, bool const lower,
                                            bool const unitsOnDiag, T* output, const sd::LongType* outputShape,
                                            const sd::LongType* tadLeftShape, const sd::LongType* tadLeftOffset,
                                            const sd::LongType* tadRightShape, const sd::LongType* tadRightOffset,
                                            const sd::LongType* tadOutputShape, const sd::LongType* tadOutputOffset,
                                            sd::LongType batchNum) {
  __shared__ sd::LongType rows;
  __shared__ sd::LongType cols;

  if (threadIdx.x == 0) {
    rows = shape::sizeAt(leftPartShape, -2);
    cols = shape::sizeAt(rightPartShape, -1);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto stop = batchNum;
  auto increment = blockDim.x * gridDim.x;

  for (auto i = start; i < stop; i += increment) {
    auto pLeftPart = leftInput + tadLeftOffset[i];
    auto pRightPart = rightInput + tadRightOffset[i];
    auto pOutputPart = output + tadOutputOffset[i];
    if (lower) {
      lowerTriangularSolve<T>(pLeftPart, tadLeftShape, pRightPart, tadRightShape, unitsOnDiag, pOutputPart,
                              tadOutputShape, rows, cols);
    } else {
      upperTriangularSolve<T>(pLeftPart, tadLeftShape, pRightPart, tadRightShape, unitsOnDiag, pOutputPart,
                              tadOutputShape, rows, cols);
    }
  }
}

template <typename T>
static sd::Status triangularSolveFunctor_(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                          bool lower, bool unitsOnDiag, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {leftInput, rightInput});

  std::vector<sd::LongType> dims = {-2, -1};
  auto leftTads = ConstantTadHelper::getInstance().tadForDimensions(leftInput->shapeInfo(), &dims);
  auto rightTads = ConstantTadHelper::getInstance().tadForDimensions(rightInput->shapeInfo(), &dims);
  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims);

  auto stream = context->getCudaStream();
  T const* leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
  T const* rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
  T* outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  triangularSolveKernel<T><<<128, 128, 256, *stream>>>(
      leftBuf, leftInput->specialShapeInfo(), rightBuf, rightInput->specialShapeInfo(), lower, unitsOnDiag, outputBuf,
      output->specialShapeInfo(), leftTads->specialShapeInfo(), leftTads->specialOffsets(), rightTads->specialShapeInfo(),
      rightTads->specialOffsets(), outputTads->specialShapeInfo(), outputTads->specialOffsets(), leftTads->numberOfTads());

  NDArray::registerSpecialUse({output}, {leftInput, rightInput});

  return sd::Status::OK;
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
void triangularSolve2D(sd::LaunchContext* context, const NDArray& leftInput, const NDArray& rightInput,
                       bool const lower, bool const unitsOnDiag, NDArray& output) {
  triangularSolveFunctor_<T>(context, const_cast<NDArray*>(&leftInput), const_cast<NDArray*>(&rightInput), lower,
                             unitsOnDiag, &output);


}
BUILD_SINGLE_TEMPLATE(template void triangularSolve2D,
                      (sd::LaunchContext * context, NDArray const& leftInput, NDArray const& rightInput,
                       bool const lower, bool const unitsOnDiag, NDArray& output),
                      SD_FLOAT_TYPES);

sd::Status triangularSolveFunctor(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool lower,
                                  bool unitsOnDiag, NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return triangularSolveFunctor_,
                        (context, leftInput, rightInput, lower, unitsOnDiag, output), SD_FLOAT_NATIVE);
}

template <typename T>
static SD_KERNEL void upperAdjointKernel(T const* input, T* output, sd::LongType batchSize, sd::LongType rows,
                                         sd::LongType columns, sd::LongType const* inputTads,
                                         sd::LongType const* inputOffsets, sd::LongType const* outputTads,
                                         sd::LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c <= r; c += blockDim.y) {
        sd::LongType zPos[] = {r, c};
        sd::LongType xPos[] = {c, r};
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(inputTads, xPos);
        outputPart[zIndex] = inputPart[xIndex];
      }
    }
  }
}

template <typename T>
static SD_KERNEL void lowerAdjointKernel(T const* input, T* output, sd::LongType batchSize, sd::LongType rows,
                                         sd::LongType columns, sd::LongType const* inputTads,
                                         sd::LongType const* inputOffsets, sd::LongType const* outputTads,
                                         sd::LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = r + threadIdx.y; c < columns; c += blockDim.y) {
        sd::LongType zPos[] = {r, c};
        sd::LongType xPos[] = {c, r};
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(inputTads, xPos);
        outputPart[zIndex] = inputPart[xIndex];
      }
    }
  }
}

template <typename T>
static void adjointTriangularMatrix_(sd::LaunchContext* context, NDArray const* input, bool const lower,
                                     NDArray* output) {
  std::vector<sd::LongType> dims = {-2, -1};
  auto inputTads = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims);
  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(),&dims);
  auto stream = context->getCudaStream();
  auto inputBuf = reinterpret_cast<T const*>(input->specialBuffer());
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);

  if (lower) {
    lowerAdjointKernel<T><<<128, 256, 256, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                      inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                      outputTads->specialShapeInfo(), outputTads->specialOffsets());
  } else {
    upperAdjointKernel<T><<<128, 256, 256, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                      inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                      outputTads->specialShapeInfo(), outputTads->specialOffsets());
  }
}

void adjointMatrix(sd::LaunchContext* context, NDArray const* input, bool const lower, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointTriangularMatrix_, (context, input, lower, output), SD_FLOAT_NATIVE);
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd
