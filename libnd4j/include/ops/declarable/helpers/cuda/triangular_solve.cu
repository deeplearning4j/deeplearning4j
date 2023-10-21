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

  printf("Entering lowerTriangularSolve\n");

  printf("Initial rows: %ld\n", rows);
  printf("Initial cols: %ld\n", cols);

  for (auto r = 0; r < rows; r++) {
    printf("Current row index: %d\n", r);

    for (auto j = 0; j < cols; j++) {
      printf("Current col index: %d\n", j);

      sd::LongType posY[] = {r, j};
      sd::LongType posX[] = {r, r};

      printf("posY array: [%ld, %ld]\n", posY[0], posY[1]);
      printf("posX array: [%ld, %ld]\n", posX[0], posX[1]);

      auto xIndex = shape::getOffset(leftInputShape, posX, 0);
      auto yIndex = shape::getOffset(rightInputShape, posY, 0);

      printf("Calculating xIndex: %ld\n", xIndex);
      printf("Calculating yIndex: %ld\n", yIndex);

      printf("lowerTriangularSolve CUDA: At (row: %d, col: %d), xIndex: %ld, yIndex: %ld\n", r, j, xIndex, yIndex);

      auto sum = rightInput[yIndex];
      printf("Fetching initial sum from rightInput: %f\n", (float)sum);

      printf("lowerTriangularSolve CUDA: Initial sum: %f\n", (float)sum);

      for (auto c = 0; c < r; c++) {
        printf("Current inner loop index: %d\n", c);

        sd::LongType pos[] = {r, c};
        sd::LongType posZCIndex[] = {c,j};

        printf("pos array for inner loop: [%ld, %ld]\n", pos[0], pos[1]);

        auto xcIndex = shape::getOffset(leftInputShape, pos, 0);
        auto zIndex = shape::getOffset(outputShape, posZCIndex, 0);

        printf("Calculating xcIndex: %ld\n", xcIndex);
        printf("Calculating zIndex: %ld\n", zIndex);

        printf("Fetching leftInput at xcIndex: %f\n", (float)leftInput[xcIndex]);
        printf("Fetching output at zIndex: %f\n", (float)output[zIndex]);

        sum -= leftInput[xcIndex] * output[zIndex];
        printf("Updated sum: %f\n", (float)sum);

        printf("lowerTriangularSolve CUDA: After iteration %d in inner loop, sum: %f\n", c, (float)sum);
      }

      auto zIndex = shape::getOffset(outputShape, posY, 0);
      printf("Calculating zIndex after inner loop: %ld\n", zIndex);

      printf("Fetching leftInput at xIndex: %f\n", (float)leftInput[xIndex]);

      output[zIndex] = unitOnDiag ? sum : sum / leftInput[xIndex];
      printf("Updating output at zIndex: %f\n", (float)output[zIndex]);

      printf("lowerTriangularSolve CUDA: Output after processing (row: %d, col: %d): %f\n", r, j, (float)output[zIndex]);
    }
  }

  printf("Exiting lowerTriangularSolve\n");
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
static SD_HOST_DEVICE void upperTriangularSolve(T const* leftInput,
                                                sd::LongType const* leftInputShape,
                                                T const* rightInput,
                                                sd::LongType const* rightInputShape,
                                                bool const unitOnDiag,
                                                T* output, const sd::LongType* outputShape,
                                                sd::LongType rows, sd::LongType cols, sd::LongType totalXLength,
                                                sd::LongType totalYLength) {

  printf("Entering upperTriangularSolve CUDA function\n");

  for (sd::LongType r = rows; r > 0; r--) {
    for (sd::LongType j = 0; j < cols; j++) {
      sd::LongType rightInputIndices[] = {r - 1, j};
      sd::LongType leftInputIndices[] = {r - 1, r - 1};

      auto xIndex = shape::getOffset(leftInputShape, leftInputIndices, 0);
      auto yIndex = shape::getOffset(rightInputShape, rightInputIndices, 0);

      auto sumBefore = rightInput[yIndex];
      printf("Initial sum for indices r-1: %lld, j: %lld is %f\n", r-1, j, static_cast<float>(sumBefore));

      auto sum = sumBefore;
      for (auto c = r; c < rows; c++) {
        sd::LongType pos[] = {r - 1, c};
        sd::LongType pos2[] = {c,j};

        auto xcIndex = shape::getOffset(leftInputShape, pos, 0);
        auto zCIndex = shape::getOffset(outputShape, pos2, 0);

        auto left_val = leftInput[xcIndex];
        auto output_val = output[zCIndex];

        sum -= left_val * output_val;
      }
      printf("Updated sum for indices r-1: %lld, j: %lld is %f\n", r-1, j, static_cast<float>(sum));

      auto zIndex = shape::getOffset(outputShape, rightInputIndices, 0);
      auto output_before = output[zIndex];
      printf("Output value before update at r-1: %lld, j: %lld is %f\n", r-1, j, static_cast<float>(output_before));

      output[zIndex] = unitOnDiag ? sum : sum / leftInput[xIndex];

      auto output_after = output[zIndex];
      printf("Output value after update at r-1: %lld, j: %lld is %f\n", r-1, j, static_cast<float>(output_after));
    }
  }

  printf("Exiting upperTriangularSolve CUDA function\n");
}









                                                                                                                                            template <typename T>
static SD_KERNEL void triangularSolveKernel(T const* leftInput,
                                            sd::LongType const* leftPartShape,
                                            T const* rightInput,
                                            sd::LongType const* rightPartShape,
                                            bool const lower,
                                            bool const unitsOnDiag,
                                            T* output, const sd::LongType* outputShape,
                                            const sd::LongType* tadLeftShape,
                                            const sd::LongType* tadLeftOffset,
                                            const sd::LongType* tadRightShape,
                                            const sd::LongType* tadRightOffset,
                                            const sd::LongType* tadOutputShape,
                                            const sd::LongType* tadOutputOffset,
                                            sd::LongType batchNum) {
  __shared__ sd::LongType rows;
  __shared__ sd::LongType cols;
  __shared__ sd::LongType xTotalLen;
  __shared__ sd::LongType yTotalLen;
  if (threadIdx.x == 0) {
    rows = shape::sizeAt(leftPartShape, -2);
    cols = shape::sizeAt(rightPartShape, -1);
    xTotalLen = shape::length(leftPartShape);
    yTotalLen = shape::length(rightPartShape);

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
                              tadOutputShape, rows, cols, xTotalLen, yTotalLen);
    }
  }
}

template <typename T>
static sd::Status triangularSolveFunctor_(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                          bool lower, bool unitsOnDiag, NDArray* output) {

  printf("CUDA: Entering triangularSolveFunctor_\n");

  NDArray::prepareSpecialUse({output}, {leftInput, rightInput});
  leftInput->printBuffer("leftInput before");
  rightInput->printBuffer("rightInput before");
  std::vector<sd::LongType> dims = {-2, -1};
  auto leftTads = ConstantTadHelper::getInstance().tadForDimensions(leftInput->shapeInfo(), &dims);
  leftTads->print("left tad:");
  auto rightTads = ConstantTadHelper::getInstance().tadForDimensions(rightInput->shapeInfo(), &dims);

  rightTads->print("right tad:");
  printf("left shape info:\n");
  shape::printShapeInfo(leftTads->primaryShapeInfo());
  printf("right shape info:\n");
  shape::printShapeInfo(rightTads->primaryShapeInfo());

  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims);
  printf("output shape info:\n");
  shape::printShapeInfo(outputTads->primaryShapeInfo());



  auto stream = context->getCudaStream();
  T const* leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
  T const* rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
  T* outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  dim3 triangularSolveDims = getLaunchDims("triangular_solve");

  printf("CUDA: Launching triangularSolveKernel\n");
  triangularSolveKernel<T><<<triangularSolveDims.y,
  triangularSolveDims.x,
  triangularSolveDims.z, *stream>>>(
      leftBuf, leftInput->specialShapeInfo(),
      rightBuf, rightInput->specialShapeInfo(),
      lower, unitsOnDiag, outputBuf,
      output->specialShapeInfo(),
      leftTads->specialShapeInfo(),
      leftTads->specialOffsets(),
      rightTads->specialShapeInfo(),
      rightTads->specialOffsets(),
      outputTads->specialShapeInfo(),
      outputTads->specialOffsets(),
      leftTads->numberOfTads());

  NDArray::registerSpecialUse({output}, {leftInput, rightInput});

  printf("CUDA: Exiting triangularSolveFunctor_\n");

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
  NDArray::prepareSpecialUse({input}, {output});
  std::vector<sd::LongType> dims = {-2, -1};
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
  } else {
    upperAdjointKernel<T><<<launchDims.y, launchDims.x,launchDims.z, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                                inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                                                outputTads->specialShapeInfo(), outputTads->specialOffsets());
  }

  NDArray::registerSpecialUse({input}, {output});
}

void adjointMatrix(sd::LaunchContext* context, NDArray const* input, bool const lower, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointTriangularMatrix_, (context, input, lower, output), SD_FLOAT_NATIVE);
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd
