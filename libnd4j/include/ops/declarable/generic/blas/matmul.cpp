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
// @author raver119@gmail.com, created on 07.10.2017.
// @author GS <sgazeos@gmail.com>, modified
// @author Yurii Shyrma (iuriish@yahoo.com), fully rewritten
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_matmul)

#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/matmul.h>
#include <ops/declarable/headers/blas.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(matmul, 2, 1, false, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  int iSize = (int)block.getIArguments()->size();
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;
  const LongType arithmetic = iSize > 3 ? INT_ARG(3) : 0;
  REQUIRE_TRUE(arithmetic == 0 || arithmetic == 1, 0,
               "MATMUL OP: arithmetic must be 0 (legacy) or 1 (SERIAL_FMA)");
  if (arithmetic == 0 && (x->isEmpty() || y->isEmpty())) return Status::OK;
  // optional use alpha nad beta
  iSize = (int)block.getTArguments()->size();
  double alpha = iSize > 0 ? T_ARG(0) : 1.0;
  double beta = iSize > 1 ? T_ARG(1) : 0.0;

  if (transZ) {
    x = INPUT_VARIABLE(1);
    y = INPUT_VARIABLE(0);
    bool temp = transX;
    transX = !transY;
    transY = !temp;
  }

  if (arithmetic == 1) {
    REQUIRE_TRUE(x->rankOf() > 0 && y->rankOf() > 0, 0,
                 "MATMUL SERIAL_FMA: scalar operands are unsupported");
    const auto outputType = block.numD() > 0 ? D_ARG(0) : x->dataType();
    REQUIRE_TRUE(block.numD() <= 1 &&
                     helpers::matmulSerialStorageSupported(x->dataType(), y->dataType(), outputType) &&
                     z->dataType() == outputType &&
                     (block.numD() > 0 || x->dataType() == y->dataType()), 0,
                 "MATMUL SERIAL_FMA: expected matching storage or explicit FLOAT output with HALF/BFLOAT16/FLOAT inputs");
    const auto expected = ShapeUtils::evalShapeForMatmul(x->shapeInfo(), y->shapeInfo(), transX, transY);
    REQUIRE_TRUE(z->isSameShape(expected), 0, "MATMUL SERIAL_FMA: output shape mismatch");
    if (x->isEmpty() || y->isEmpty()) return Status::OK;
    REQUIRE_TRUE(x->getDataBuffer() != z->getDataBuffer() && y->getDataBuffer() != z->getDataBuffer(), 0,
                 "MATMUL SERIAL_FMA: output must not alias an input");
    // Preserve the logical strides instead of flattening batch/transposed axes.
#if defined(SD_VULKAN) || defined(SD_TPU)
    REQUIRE_TRUE(false, 0, "MATMUL SERIAL_FMA: no native recipe for this artifact");
#else
    MmulHelper::matmulSerial(block.launchContext(), x, y, z, transX, transY, alpha, beta);
#endif
    return Status::OK;
  }

  // Compute ranks AFTER potential transZ swap
  const int xRank = x->rankOf();
  const int yRank = y->rankOf();
  const int zRank = z->rankOf();

  const int xLastDim = transX ? -2 : -1;
  const int yLastDim = transY ? -2 : -1;
  const int xLastButOneDim = transX ? -1 : -2;
  const int yLastButOneDim = transY ? -1 : -2;

  // ******* input validation ******* //
  REQUIRE_TRUE(xRank > 0 && yRank > 0, 0,
               "MATMUL OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank "
               "= %i, y rank = %i !",
               xRank, yRank);

  if (xRank == 1 && yRank == 1) {  // dot case, output is scalar (or vector with length = 1)
    REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,
                 "MATMUL OP: since input arrays are vectors they must have the same length, but got x length = %i, y "
                 "length = %i !",
                 x->lengthOf(), y->lengthOf());
  } else if (xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
    REQUIRE_TRUE(x->lengthOf() == y->sizeAt(yLastButOneDim), 0,
                 "MATMUL OP: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
  } else if (xRank == 2 && yRank == 1) {  // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
    REQUIRE_TRUE(x->sizeAt(xLastDim) == y->lengthOf(), 0,
                 "MATMUL OP: input arrays have inconsistent shapes for matrix-vector product: x %s, y %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
  } else if (xRank != yRank) {  // ONNX MatMul broadcast, i.e. [2,3,4] x [4,5] = [2,3,5] or [3,4] x [2,4,5] = [2,3,5]
    NDArray* batched = xRank > yRank ? x : y;
    REQUIRE_TRUE((xRank == 2 || yRank == 2) && batched->rankOf() > 2 && zRank == batched->rankOf(), 0,
                 "MATMUL OP: inputs of different ranks need one 2D input and an output with the other input's rank, "
                 "but got instead: x rank = %i, y rank = %i, z rank = %i !",
                 xRank, yRank, zRank);
    REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) && x->sizeAt(xLastButOneDim) == z->sizeAt(-2) &&
                 y->sizeAt(yLastDim) == z->sizeAt(-1),
                 0, "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                 ShapeUtils::shapeAsString(z).c_str());
    for (int i = 0; i < zRank - 2; ++i)
      REQUIRE_TRUE(batched->sizeAt(i) == z->sizeAt(i), 0,
                   "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                   ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                   ShapeUtils::shapeAsString(z).c_str());
  } else {
    REQUIRE_TRUE(xRank == zRank, 0,
                 "MATMUL OP: input and output arrays must have the same rank, but got instead: x rank = %i, y rank = "
                 "%i, z rank = %i !",
                 xRank, yRank, zRank);
    REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) && x->sizeAt(xLastButOneDim) == z->sizeAt(-2) &&
                 y->sizeAt(yLastDim) == z->sizeAt(-1),
                 0, "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                 ShapeUtils::shapeAsString(z).c_str());

    if (xRank > 2)  // outer dims must be the same
      for (int i = 0; i < xRank - 2; ++i)
    REQUIRE_TRUE(x->sizeAt(i) == y->sizeAt(i) && y->sizeAt(i) == z->sizeAt(i), 0,
                 "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                 ShapeUtils::shapeAsString(z).c_str());
  }
  // ******* end of input validation ******* //

  if (xRank > 2 && yRank == 2 && !transX) {
    // Every row of x meets the same y, so x's batch folds into the rows of one 2D product.
    // x and z must fold their rows in the same order, so both are reshaped in logical 'c'
    // order; reshape returns a view when the strides allow one and a copy otherwise.
    std::vector<LongType> xFoldedShape = {x->lengthOf() / x->sizeAt(-1), x->sizeAt(-1)};
    NDArray* xFolded = x->reshape('c', xFoldedShape, false);
    std::vector<LongType> zFoldedShape = {z->lengthOf() / z->sizeAt(-1), z->sizeAt(-1)};
    NDArray* zFolded = z->reshape('c', zFoldedShape, false);

    MmulHelper::matmul(xFolded, y, zFolded, transX, transY, alpha, beta);

    const bool zCopied = zFolded->getDataBuffer() != z->getDataBuffer();
    if (zCopied) {
      // z has no 'c' view of its folded rows, so the product went to a copy. Write it back
      // through a view of the copy in z's own shape, which keeps the row order.
      std::vector<LongType>* zShape = z->getShapeAsVector();
      NDArray* unfolded = zFolded->reshape('c', *zShape, false);
      delete zShape;
      z->assign(unfolded);
      delete unfolded;
    }
    // Copies are retired behind the stream that still reads them; views only drop their wrapper.
    if (xFolded->getDataBuffer() != x->getDataBuffer())
      MmulHelper::deleteTemporary(xFolded);
    else
      delete xFolded;
    if (zCopied)
      MmulHelper::deleteTemporary(zFolded);
    else
      delete zFolded;
    return Status::OK;
  }

  // A transposed batched x, or a batched y, keeps its shape: MmulHelper::matmul transposes
  // each operand by its own rank and mmulNxN pairs the 2D operand with every batch.
  MmulHelper::matmul(x, y, z, transX, transY, alpha, beta);

  return Status::OK;
}

DECLARE_SYN(mMul, matmul);

DECLARE_SYN(mmul, matmul);

DECLARE_SYN(gemm, matmul);

DECLARE_SYN(gemv, matmul);

DECLARE_SYN(dot, matmul);

//////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(matmul) {
  auto xShapeInfo = inputShape->at(0);
  auto yShapeInfo = inputShape->at(1);


  const int iSize = (int)block.getIArguments()->size();
  REQUIRE_TRUE(iSize < 4 || INT_ARG(3) == 0 || INT_ARG(3) == 1, 0,
               "MATMUL OP: invalid arithmetic contract");
  if (iSize > 3 && INT_ARG(3) == 1) {
    REQUIRE_TRUE(shape::rank(xShapeInfo) > 0 && shape::rank(yShapeInfo) > 0, 0,
                 "MATMUL SERIAL_FMA: scalar operands are unsupported");
    const auto dtype = ArrayOptions::dataType(xShapeInfo);
    const auto otherType = ArrayOptions::dataType(yShapeInfo);
    const auto outputType = block.numD() > 0 ? D_ARG(0) : dtype;
    REQUIRE_TRUE(block.numD() <= 1 && helpers::matmulSerialStorageSupported(dtype, otherType, outputType) &&
                     (block.numD() > 0 || dtype == otherType), 0,
                 "MATMUL SERIAL_FMA: expected matching storage or explicit FLOAT output with HALF/BFLOAT16/FLOAT inputs");
  }
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;

  if (transZ) {
    xShapeInfo = inputShape->at(1);
    yShapeInfo = inputShape->at(0);
    bool temp = transX;
    transX = !transY;
    transY = !temp;
  }

  auto zShapeOnly = ShapeUtils::evalShapeForMatmul(xShapeInfo, yShapeInfo, transX, transY);

  auto dtypeX = ArrayOptions::dataType(xShapeInfo);
  auto dtypeY = ArrayOptions::dataType(yShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto yOrder = shape::order(yShapeInfo);
  auto zOrder = xOrder == 'c' && yOrder == 'c' ? 'c' : 'f';

  // The enum order put BFLOAT16 and every integer type above FLOAT32 and DOUBLE.
  auto dtypeZ = helpers::matmulOutputType(dtypeX, dtypeY);
  if (iSize > 3 && INT_ARG(3) == 1 && block.numD() > 0) dtypeZ = D_ARG(0);
  if(shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) {
    const auto emptyType = iSize > 3 && INT_ARG(3) == 1 ? dtypeZ : dtypeX;
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(emptyType, zShapeOnly));
  }

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtypeZ, zOrder, zShapeOnly);
  return SHAPELIST(newShape);
}

//////////////////////////////////////////////////////////////////////
DECLARE_TYPES(matmul) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS, ALL_INTS});
}

//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(matmul_bp, 3, 2, false, 0, -2) {
  REQUIRE_TRUE(block.getIArguments()->size() < 4 || INT_ARG(3) == 0, 0,
               "MATMUL_BP: SERIAL_FMA is an inference-only arithmetic contract");
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto eps = INPUT_VARIABLE(2);
  auto dldx = OUTPUT_VARIABLE(0);
  auto dldy = OUTPUT_VARIABLE(1);

  int iSize = (int)block.getIArguments()->size();
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;

  // Optional alpha and beta mirror matmul's. The gradient of alpha * op(x) * op(y) + beta * z scales
  // by alpha and has no beta term, so beta never reaches the gradient products.
  iSize = (int)block.getTArguments()->size();
  double alpha = iSize > 0 ? T_ARG(0) : 1.0;

  /*
  In: x=[a,b], y=[b,c]
  tX  tY  tZ  x       y       z       dz          dLdx                                    dLdy
  F   F   F   [a,b]   [b,c]   [a,c]   [a,c]       [a,c]*[b,c]T = [a,b]        x*yT        [a,b]T*[a,c] = [b,c] xT*y T F
  F   [b,a]   [b,c]   [a,c]   [a,c]       ([a,c]*[b,c]T)T = [b,a]     (x*yT)T     [b,a]*[a,c] = [b,c]         x*y F   T
  F   [a,b]   [c,b]   [a,c]   [a,c]       ([a,c]*[c,b]) = [a,b]       x*y         [a,b]T*[a,c] = [b,c] ->T    xT*y T   T
  F   [b,a]   [c,b]   [a,c]   [a,c]       ([a,c]*[c,b])T = [b,a]      (x*y)T      [b,a]*[a,c] = [b,c]  ->T    x*y F   F
  T   [a,b]   [b,c]   [c,a]   [c,a]
  */
  // special case for scalar value
  if (eps->isScalar()) {
    if (x->isVector() && y->isVector()) {
      // A scalar product of two vectors is alpha * sum_k x_k * y_k whatever their orientations,
      // so each gradient is the other vector times alpha * eps, element by element.
      NDArray *dldxTemp = (*eps) * (*y);
      if (alpha != 1.0) *dldxTemp *= alpha;
      dldx->assign(dldxTemp);
      delete dldxTemp;
      NDArray *dldyTemp = (*eps) * (*x);
      if (alpha != 1.0) *dldyTemp *= alpha;
      dldy->assign(dldyTemp);
      delete dldyTemp;
    } else {
      dldx->assign(alpha);
      dldy->assign(alpha);
      
      // match the dimensions for reduction for matrix multiply: columns on first input, rows on second input
      // the dimensions should match the matching dimensions to compute proper gradients wrt each input
      // core gradient for each is sum(input) * eps as scalar
      std::vector<LongType> axesZero({0});
      NDArray *xSum = x->reduceAlongDimension(reduce::Sum, &axesZero);
      NDArray *xSumScaled = *xSum * (*eps);
      std::vector<sd::LongType> xSumShape = {xSumScaled->lengthOf(), 1};
      NDArray* xSumRow = xSumScaled->reshape(xSumScaled->ordering(), xSumShape);
      
      std::vector<LongType> axes({1});
      NDArray *ySum = y->reduceAlongDimension(reduce::Sum, &axes);
      NDArray *ySumScaled = *ySum * (*eps);
      std::vector<sd::LongType> ySumShape = {1, ySumScaled->lengthOf()};
      NDArray* ySumRow = ySumScaled->reshape(ySumScaled->ordering(), ySumShape);

      // execute proper multiplication: rows for first input, columns for second
      dldx->mulRowVector(ySumRow, dldx);
      dldy->muliColumnVector(xSumRow);

      // FIXED: Proper cleanup - delete each allocated array once, add missing cleanup
      delete xSumRow;
      delete xSumScaled;
      delete xSum;
      delete ySumRow;
      delete ySumScaled;
      delete ySum;
    }

    return Status::OK;
  }

  matmul op;
  // Forward: Z = op(X, transX) * op(Y, transY), optionally transposed by transZ
  // Backward: the correct general formulas that handle all 8 combinations of transX/transY/transZ:
  //   dL/dX = matmul(eps, Y, transZ, !transY, transX)
  //   dL/dY = matmul(X, eps, !transX, transZ, transY)
  // With inputs of different ranks the 2D input met every batch of the other input, so its
  // gradient sums over the batch. Each factor of that sum is materialized with its batch folded
  // into rows in the same 'c' order, and the sum is one flat^T * flat product.
  const int xRank = x->rankOf();
  const int yRank = y->rankOf();
  auto lastTwoSwapped = [](NDArray* array) {
    const int rank = array->rankOf();
    std::vector<LongType> permut(rank);
    for (int i = 0; i < rank - 2; ++i) permut[i] = i;
    permut[rank - 2] = rank - 1;
    permut[rank - 1] = rank - 2;
    NDArray* view = array->permute(permut, false, false);
    NDArray* materialized = view->dup('c');
    delete view;
    return materialized;
  };
  auto foldRows = [](NDArray* array) {
    LongType rows = 1;
    for (int i = 0; i < array->rankOf() - 1; ++i) rows *= array->sizeAt(i);
    std::vector<LongType> folded = {rows, array->sizeAt(-1)};
    return array->reshape('c', folded, false);
  };
  // Copies are retired behind the stream that still reads them; views only drop their wrapper.
  auto release = [](NDArray* derived, NDArray* source) {
    if (derived == source) return;
    if (derived->getDataBuffer() != source->getDataBuffer())
      MmulHelper::deleteTemporary(derived);
    else
      delete derived;
  };
  // Multiplies a^T * b into out when the 2D input is not transposed, else b^T * a.
  auto batchSummedGradient = [&](NDArray* a, NDArray* b, NDArray* out, const bool transOut) {
    NDArray* aFlat = foldRows(a);
    NDArray* bFlat = foldRows(b);
    if (transOut)
      MmulHelper::matmul(bFlat, aFlat, out, true, false, alpha, 0.0);
    else
      MmulHelper::matmul(aFlat, bFlat, out, true, false, alpha, 0.0);
    release(aFlat, a);
    release(bFlat, b);
  };

  if (xRank == 2 && yRank > 2) {
    // dL/dop(x) = sum over batches of G * op(y)^T with G = transZ ? eps^T : eps; the batch
    // folds into the rows of G^T and op(y)^T.
    NDArray* gradT = transZ ? eps : lastTwoSwapped(eps);
    NDArray* opYT = transY ? y : lastTwoSwapped(y);
    batchSummedGradient(gradT, opYT, dldx, transX);
    release(gradT, eps);
    release(opYT, y);
  } else {
    op.execute({eps, y}, {dldx}, {alpha, 0.0}, {transZ, transY ? 0 : 1, transX}, {});
  }

  if (xRank > 2 && yRank == 2) {
    // dL/dop(y) = sum over batches of op(x)^T * G with G = transZ ? eps^T : eps; the batch
    // folds into the rows of op(x) and G.
    NDArray* opX = transX ? lastTwoSwapped(x) : x;
    NDArray* grad = transZ ? lastTwoSwapped(eps) : eps;
    batchSummedGradient(opX, grad, dldy, transY);
    release(opX, x);
    release(grad, eps);
  } else {
    op.execute({x, eps}, {dldy}, {alpha, 0.0}, {transX ? 0 : 1, transZ, transY}, {});
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(matmul_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

//////////////////////////////////////////////////////////////////////
DECLARE_TYPES(matmul_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(1, {ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif
