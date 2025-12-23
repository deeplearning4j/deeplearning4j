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
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(matmul, 2, 1, false, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  if(x->isEmpty() || y->isEmpty())
    return Status::OK;
  int iSize = (int)block.getIArguments()->size();
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;
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

  // Handle rank mismatch when one input has singleton leading dimensions
  // This supports ONNX Gemm patterns like [1,1,1,768] x [768,768] -> [1,1,1,768]
  NDArray* xReshaped = nullptr;
  NDArray* yReshaped = nullptr;
  NDArray* zReshaped = nullptr;

  if (xRank != yRank && xRank > 2 && yRank == 2) {
    // Check if x has all singleton leading dims
    bool allLeadingSingleton = true;
    for (int i = 0; i < xRank - 2; ++i) {
      if (x->sizeAt(i) != 1) {
        allLeadingSingleton = false;
        break;
      }
    }
    if (allLeadingSingleton) {
      // Reshape x from [1,1,...,M,K] to [M,K] for matmul
      std::vector<LongType> newXShape = {x->sizeAt(-2), x->sizeAt(-1)};
      xReshaped = new NDArray(x->reshape(x->ordering(), newXShape));
      // Reshape z from [1,1,...,M,N] to [M,N]
      std::vector<LongType> newZShape = {z->sizeAt(-2), z->sizeAt(-1)};
      zReshaped = new NDArray(z->reshape(z->ordering(), newZShape));
      x = xReshaped;
      z = zReshaped;
    }
  } else if (xRank != yRank && yRank > 2 && xRank == 2) {
    // Check if y has all singleton leading dims
    bool allLeadingSingleton = true;
    for (int i = 0; i < yRank - 2; ++i) {
      if (y->sizeAt(i) != 1) {
        allLeadingSingleton = false;
        break;
      }
    }
    if (allLeadingSingleton) {
      // Reshape y from [1,1,...,K,N] to [K,N] for matmul
      std::vector<LongType> newYShape = {y->sizeAt(-2), y->sizeAt(-1)};
      yReshaped = new NDArray(y->reshape(y->ordering(), newYShape));
      // Reshape z from [1,1,...,M,N] to [M,N]
      std::vector<LongType> newZShape = {z->sizeAt(-2), z->sizeAt(-1)};
      zReshaped = new NDArray(z->reshape(z->ordering(), newZShape));
      y = yReshaped;
      z = zReshaped;
    }
  }

  // Update ranks after potential reshaping
  const int xRankFinal = x->rankOf();
  const int yRankFinal = y->rankOf();
  const int zRankFinal = z->rankOf();

  if (xRankFinal == 1 && yRankFinal == 1) {  // dot case, output is scalar (or vector with length = 1)
    REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,
                 "MATMUL OP: since input arrays are vectors they must have the same length, but got x length = %i, y "
                 "length = %i !",
                 x->lengthOf(), y->lengthOf());
  } else if (xRankFinal == 1 && yRankFinal == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
    REQUIRE_TRUE(x->lengthOf() == y->sizeAt(yLastButOneDim), 0,
                 "MATMUL OP: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
  } else if (xRankFinal == 2 && yRankFinal == 1) {  // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
    REQUIRE_TRUE(x->sizeAt(xLastDim) == y->lengthOf(), 0,
                 "MATMUL OP: input arrays have inconsistent shapes for matrix-vector product: x %s, y %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
  } else {
    REQUIRE_TRUE(xRankFinal == yRankFinal && yRankFinal == zRankFinal, 0,
                 "MATMUL OP: input and output arrays must have the same rank, but got instead: x rank = %i, y rank = "
                 "%i, z rank = %i !",
                 xRankFinal, yRankFinal, zRankFinal);
    REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) && x->sizeAt(xLastButOneDim) == z->sizeAt(-2) &&
                 y->sizeAt(yLastDim) == z->sizeAt(-1),
                 0, "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                 ShapeUtils::shapeAsString(z).c_str());

    if (xRankFinal > 2)  // outer dims must be the same
      for (int i = 0; i < xRankFinal - 2; ++i)
    REQUIRE_TRUE(x->sizeAt(i) == y->sizeAt(i) && y->sizeAt(i) == z->sizeAt(i), 0,
                 "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                 ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                 ShapeUtils::shapeAsString(z).c_str());
  }
  // ******* end of input validation ******* //

  MmulHelper::matmul(x, y, z, transX, transY, alpha, beta, z);

  // Clean up reshaped arrays
  delete xReshaped;
  delete yReshaped;
  delete zReshaped;

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

  // we just pick the higher data type out of X and Y
  auto dtypeZ = dtypeX > dtypeY ? dtypeX : dtypeY;
  if(shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) {
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(xShapeInfo),zShapeOnly));
  }

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtypeZ, zOrder, zShapeOnly);
  return SHAPELIST(newShape);
}

//////////////////////////////////////////////////////////////////////
DECLARE_TYPES(matmul) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS, ALL_INTS});
}

//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(matmul_bp, 3, 2, false, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto eps = INPUT_VARIABLE(2);
  auto dldx = OUTPUT_VARIABLE(0);
  auto dldy = OUTPUT_VARIABLE(1);

  int iSize = (int)block.getIArguments()->size();
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;

  // optional use alpha nad beta
  iSize = (int)block.getTArguments()->size();

  double alpha = iSize > 0 ? T_ARG(0) : 1.0;
  double beta = iSize > 1 ? T_ARG(1) : 0.0;

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
      if (x->isRowVector() && y->isRowVector()) {
        float ySum = y->sumNumber().e<float>(0);
        NDArray *dldxTemp = (*eps) * ySum;
        dldx->assign(dldxTemp);
        delete dldxTemp;

        float xSum = x->sumNumber().e<float>(0);
        NDArray *dldyTemp = (*eps) * xSum;
        dldy->assign(dldyTemp);
        delete dldyTemp;
      } else if (x->isColumnVector() && y->isColumnVector()) {
        float ySum = y->sumNumber().e<float>(0);
        NDArray *dldxTemp = (*eps) * ySum;
        dldx->assign(dldxTemp);
        delete dldxTemp;
        float xSum = x->sumNumber().e<float>(0);
        NDArray *dldyTemp = (*eps) * xSum;
        dldy->assign(dldyTemp);
        delete dldyTemp;
      } else {
        NDArray *dldxTemp = (*eps) * (*y);
        dldx->assign(dldxTemp);
        delete dldxTemp;
        NDArray *dldyTemp = (*eps) * (*x);
        dldy->assign(dldyTemp);
        delete dldyTemp;
      }
    } else {
      // assign all ones to shape as baseline
      auto alphaBetaBase = 1.0f;
      if (alpha > 0.0f) {
        alphaBetaBase *= alpha;
      }

      if (beta > 0.0f) {
        alphaBetaBase += beta;
      }

      dldx->assign(alphaBetaBase);
      dldy->assign(alphaBetaBase);
      
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
  op.execute({eps, y}, {dldx}, {alpha, beta}, {transZ, !transY, transX}, {});
  op.execute({x, eps}, {dldy}, {alpha, beta}, {!transX, transZ, transY}, {});

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
}

}  // namespace ops
}  // namespace sd

#endif
