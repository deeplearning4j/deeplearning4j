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
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/lstsq.h>
#include <ops/declarable/helpers/lup.h>
#include <ops/declarable/helpers/qr.h>
#include <ops/declarable/helpers/triangular_solve.h>
#include <system/op_boilerplate.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void fillRegularizerKernel(T* ioMatrixData, const LongType* ioMatrixShape,
                                            const LongType* ioMatrixTads, const LongType* ioMatrixOffsets,
                                            LongType batchSize, LongType rows, T const value) {
  for (auto x = blockIdx.x; x < batchSize; x += gridDim.x) {
    auto z = ioMatrixData + ioMatrixOffsets[x];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      LongType pos[] = {r, r};
      auto zIndex = shape::getOffset(ioMatrixTads, pos);
      z[zIndex] = value;
    }
  }
}

template <typename T>
static void fillRegularizer(LaunchContext* context, NDArray& ioMatrix, double const value) {
  std::vector<LongType> dims = {-2, -1};
  auto lastDimsTads = ConstantTadHelper::getInstance().tadForDimensions(ioMatrix.shapeInfo(), &dims);
  auto stream = context->getCudaStream();
  auto rows = ioMatrix.sizeAt(-2);
  dim3 launchDims = getLaunchDims("lstsq_reg");
  fillRegularizerKernel<T><<<launchDims.y,launchDims.x,launchDims.z, *stream>>>(
      ioMatrix.dataBuffer()->specialAsT<T>(), ioMatrix.specialShapeInfo(), lastDimsTads->specialShapeInfo(),
      lastDimsTads->specialOffsets(), lastDimsTads->numberOfTads(), rows, (T)value);

}

template <typename T>
Status leastSquaresSolveFunctor_(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                 double const l2Regularizer, bool const fast, NDArray* output) {
  if (fast) {  // Cholesky decomposition approach
    // Equation for solve A^T * Ax = A^T * b, so
    // 1. Computing A2:
    auto tAtShape = ShapeUtils::evalShapeForMatmul(leftInput->shapeInfo(), leftInput->shapeInfo(), true, false);
    // tAtShape[tAtShape.size() - 2] = output->sizeAt(-2);
    NDArray leftOutput(leftInput->ordering(), tAtShape, output->dataType(), context);
    MmulHelper::matmul(leftInput, leftInput, &leftOutput, true, false,1.0,0.0,&leftOutput);  // Computing A2 = A^T * A
    // 2. Computing B' = A^T * b
    auto rightOutput = output->ulike();

    MmulHelper::matmul(leftInput, rightInput, &rightOutput, true, false,1.0,0.0,&rightOutput);  // Computing B' = A^T * b
    // 3. Regularization ( indeed A' = A2 - l2Regularizer * I)
    if (l2Regularizer != 0.0) {
      auto regularizer = leftOutput.ulike();
      regularizer.nullify();
      fillRegularizer<T>(context, regularizer, (T)l2Regularizer);
      leftOutput += regularizer;
    }

    // 4. Cholesky decomposition -- output matrix is square and lower triangular
    cholesky(context, &leftOutput, &leftOutput, true);  // inplace decomposition
    // 5. Solve two triangular systems:
    auto rightB = rightOutput.ulike();
    rightB.nullify();

    triangularSolveFunctor(context, &leftOutput, &rightOutput, true, false, &rightB);

    adjointMatrix(context, &leftOutput, true, &leftOutput);
    triangularSolveFunctor(context, &leftOutput, &rightB, false, false, output);
    // All done
  } else {  // QR decomposition approach
    // Equation for solve Rx = Q^T * b, where A = Q * R, where Q - orthogonal matrix, and R - upper triangular
    // 1. QR decomposition
    auto qShape = leftInput->getShapeAsVector();
    auto rShape = leftInput->getShapeAsVector();
    qShape[leftInput->rankOf() - 1] = leftInput->sizeAt(-2);

    NDArray Q(leftInput->ordering(), qShape, leftInput->dataType(), context);
    NDArray R(leftInput->ordering(), rShape, leftInput->dataType(), context);
    qr(context, leftInput, &Q, &R, true);
    // 2. b` = Q^t * b:
    auto rightOutput = rightInput->ulike();
    MmulHelper::matmul(&Q, rightInput, &rightOutput, true, false,1.0,0.0,&rightOutput);
    // 3. Solve triangular system
    triangularSolveFunctor(context, &R, &rightOutput, false, false, output);
  }
  return Status::OK;
}

Status leastSquaresSolveFunctor(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                    double const l2Regularizer, bool const fast, NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return leastSquaresSolveFunctor_,
                        (context, leftInput, rightInput, l2Regularizer, fast, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
