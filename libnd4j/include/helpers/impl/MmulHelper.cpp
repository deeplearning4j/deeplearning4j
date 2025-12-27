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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_CPP
#define LIBND4J_MMULHELPER_CPP
#include "../MmulHelper.h"

#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/shape.h>
#include <ops/declarable/helpers/batched_gemm.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "ops/declarable/headers/blas.h"

namespace sd {

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* A, NDArray* B,
                               const std::initializer_list<LongType>& axesA,
                               const std::initializer_list<LongType>& axesB) {
  std::vector<LongType> aA(axesA);
  std::vector<LongType> aB(axesB);
  return tensorDot(A, B, aA, aB);
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* A, NDArray* B, const std::vector<LongType>& axesA,
                               const std::vector<LongType>& axesB) {
  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;

  auto outShape = ShapeUtils::evalShapeForTensorDot(A, B, axesA, axesB, permutAt, permutBt, shapeAt, shapeBt);

  // check whether permutation is necessary
  NDArray* aP = permutAt.empty() ? A : A->permute(permutAt, false, false);
  NDArray* bP = permutBt.empty() ? B : B->permute(permutBt, false, false);

  // check whether reshape is necessary
  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : aP->reshape(aP->ordering(), shapeAt);
  NDArray* bPR = bP->isSameShape(shapeAt) ? bP : bP->reshape(bP->ordering(), shapeBt);

  NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);

  c->reshapei(outShape);

  // Delete reshaped arrays first
  if(aPR != A && aPR != aP) {
    delete aPR;
  }
  if(bPR != B && bPR != bP) {
    delete bPR;
  }

  // Then delete permuted arrays
  if(aP != A) {
    delete aP;
  }
  if(bP != B) {
    delete bP;
  }

  return c;
}


void MmulHelper::computeNewShapesAndAxes(
    NDArray& as_, const std::vector<LongType>& axes_a,
    NDArray& bs, const std::vector<LongType>& axes_b,
    std::vector<LongType>& newshape_a, std::vector<LongType>& newaxes_a,
    std::vector<LongType>& newshape_b, std::vector<LongType>& newaxes_b
) {
  // Use rankOf() and sizeAt() directly to avoid getShapeAsVector allocation
  const int aRank = as_.rankOf();
  const int bRank = bs.rankOf();

  std::vector<LongType> notin_a;
  for(int k = 0; k < aRank; ++k) {
    if(std::find(axes_a.begin(), axes_a.end(), k) == axes_a.end())
      notin_a.push_back(k);
  }

  newaxes_a.clear();
  std::copy(notin_a.begin(), notin_a.end(), std::back_inserter(newaxes_a));
  std::copy(axes_a.begin(), axes_a.end(), std::back_inserter(newaxes_a));

  LongType N2_a = std::accumulate(axes_a.begin(), axes_a.end(), 1L, [&](LongType product, LongType i){
    return product * as_.sizeAt(i);
  });

  newshape_a.clear();
  newshape_a.push_back(std::accumulate(notin_a.begin(), notin_a.end(), 1L, [&](LongType product, LongType i){
    return product * as_.sizeAt(i);
  }));
  newshape_a.push_back(N2_a);

  std::vector<LongType> notin_b;
  for(int k = 0; k < bRank; ++k) {
    if(std::find(axes_b.begin(), axes_b.end(), k) == axes_b.end())
      notin_b.push_back(k);
  }

  newaxes_b.clear();
  std::copy(axes_b.begin(), axes_b.end(), std::back_inserter(newaxes_b));
  std::copy(notin_b.begin(), notin_b.end(), std::back_inserter(newaxes_b));

  LongType N2_b = std::accumulate(axes_b.begin(), axes_b.end(), 1L, [&](LongType product, LongType i){
    return product * bs.sizeAt(i);
  });

  newshape_b.clear();
  newshape_b.push_back(N2_b);
  newshape_b.push_back(std::accumulate(notin_b.begin(), notin_b.end(), 1L, [&](LongType product, LongType i){
    return product * bs.sizeAt(i);
  }));
}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot2(NDArray* a, NDArray* b, NDArray* c, const std::vector<LongType>& axes_a,
                            const std::vector<LongType>& axes_b, std::vector<LongType>& permutAt,
                            std::vector<LongType>& permuteBt, std::vector<LongType>& permuteCt,
                            NDArray* realFinalResult) {

  // check whether permutation is required
  NDArray* cP = permuteCt.empty() ? c : c->permute(permuteCt, false, false);

  std::vector<LongType> newshape_a, newaxes_a, newshape_b, newaxes_b;
  computeNewShapesAndAxes(*a, axes_a, *b, axes_b, newshape_a, newaxes_a, newshape_b, newaxes_b);

  NDArray* aP = permutAt.empty() ? a : a->permute(permutAt, false, false);
  NDArray* bP = permuteBt.empty() ? b : b->permute(permuteBt, false, false);

  // Try view first, only copy if needed for contiguity
  NDArray* aPermuted = aP->permute(newaxes_a, false, false);
  NDArray* aPR = aPermuted->reshape('c', newshape_a, false);
  // If reshape couldn't create view (non-contiguous), need to dup
  if (aPR == nullptr || (!aPR->isView() && aPR->buffer() != aPermuted->buffer())) {
    if (aPR != nullptr && aPR != aPermuted) delete aPR;
    aPR = aPermuted->reshape('c', newshape_a, true);
  }

  NDArray* bPermuted = bP->permute(newaxes_b, false, false);
  NDArray* bPR = bPermuted->reshape('c', newshape_b, false);
  if (bPR == nullptr || (!bPR->isView() && bPR->buffer() != bPermuted->buffer())) {
    if (bPR != nullptr && bPR != bPermuted) delete bPR;
    bPR = bPermuted->reshape('c', newshape_b, true);
  }

  std::vector<LongType> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};
  NDArray* cPR = cP->reshape('f', requiredCshape, false);

  mmul(aPR, bPR, cPR, 1.0, 0.0);

  // Copy result back if buffers differ
  if (cPR->buffer() != c->buffer()) {
    c->assign(cPR);
  }

  if (realFinalResult != nullptr && realFinalResult != c) {
    realFinalResult->assign(c);
  }

  // Cleanup in reverse order of creation
  if (aPR != aPermuted && !aPR->isView()) delete aPR;
  if (aPermuted != aP && !aPermuted->isView()) delete aPermuted;
  if (aP != a && !aP->isView()) delete aP;

  if (bPR != bPermuted && !bPR->isView()) delete bPR;
  if (bPermuted != bP && !bPermuted->isView()) delete bPermuted;
  if (bP != b && !bP->isView()) delete bP;

  if (cPR != cP && !cPR->isView()) delete cPR;
  if (cP != c && !cP->isView()) delete cP;
}


void MmulHelper::tensorDot(NDArray* a, NDArray* b, NDArray* c,
                           std::vector<LongType>& axes_a, std::vector<LongType>& axes_b,
                           std::vector<LongType>& permutForC) {

  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;
  ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);


  // check whether permutation is required - use view (no copy)
  NDArray* cP = permutForC.empty() ? c : c->permute(permutForC, false, false);
  // check whether permutation is necessary - use view (no copy)
  NDArray* aP = permutAt.empty() ? a : a->permute(permutAt, false, false);
  NDArray* bP = permutBt.empty() ? b : b->permute(permutBt, false, false);

  // check whether reshape is necessary - use copyToNewBuff=false to avoid copies when possible
  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : aP->reshape(aP->ordering(), shapeAt, false);
  NDArray* bPR = bP->isSameShape(shapeBt) ? bP : bP->reshape(bP->ordering(), shapeBt, false);

  std::vector<LongType> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};

  NDArray* cPR = cP->isSameShape(requiredCshape) ? cP : cP->reshape(cP->ordering(), requiredCshape, false);
  mmul(aPR, bPR, cPR, 1.0, 0.0);

  // Only copy if cPR doesn't share buffer with c (meaning reshape created a new buffer)
  if (cPR->buffer() != c->buffer()) {
    c->assign(cPR);
  }

  // Cleanup - delete non-view arrays that were created
  if (aPR != aP && !aPR->isView()) delete aPR;
  if (bPR != bP && !bPR->isView()) delete bPR;
  if (cPR != cP && !cPR->isView()) delete cPR;
  if (aP != a && !aP->isView()) delete aP;
  if (bP != b && !bP->isView()) delete bP;
  if (cP != c && !cP->isView()) delete cP;
}

#ifndef __JAVACPP_HACK__
//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot(NDArray* a, NDArray* b, NDArray* c,
                           std::vector<std::vector<LongType>>& modifA,
                           std::vector<std::vector<LongType>>& modifB,
                           std::vector<std::vector<LongType>>& modifC) {
  NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
  std::string whatToDoWithA, whatToDoWithB,
      whatToDoWithC;  // "" - nothing; "p" - permutation; "r" - reshaping; "pr" - permutation+reshaping; "rp" -
  // reshaping/permutation, and so on; if another string is produced - throw exception

  for (const auto& arr : modifA)
    whatToDoWithA =
        (std::find(arr.begin(), arr.end(), 0) != arr.end())
        ? whatToDoWithA + "p"
        : whatToDoWithA +
          "r";  // when 0 is present in arr then it is permutation array, otherwise - it is reshaping array
  for (const auto& arr : modifB)
    whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithB + "p" : whatToDoWithB + "r";
  for (const auto& arr : modifC)
    whatToDoWithC = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithC + "p" : whatToDoWithC + "r";

  // first step for a array - use view (no copy) when possible

  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0], false, false)
                                    : a->reshape(a->ordering(), modifA[0], false);
  // first step for b array - use view (no copy) when possible
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0], false, false)
                                    : b->reshape(b->ordering(), modifB[0], false);
  // rest steps for a array
  for (size_t i = 1; i < whatToDoWithA.size(); ++i)
    if (whatToDoWithA[i] == 'p')
      aPR->permutei(modifA[i], false, false);
    else
      aPR->reshapei(modifA[i]);
  // rest steps for b array
  for (size_t i = 1; i < whatToDoWithB.size(); ++i)
    if (whatToDoWithB[i] == 'p')
      bPR->permutei(modifB[i], false, false);
    else
      bPR->reshapei(modifB[i]);

  // now work with c array
  std::vector<NDArray*> cArrs = {c};
  if (!whatToDoWithC.empty()) {
    cArrs = std::vector<NDArray*>(whatToDoWithC.size() + 1, c);
    for (size_t i = 0; i < cArrs.size() - 1; ++i)
      cArrs[i + 1] =
          (whatToDoWithC[i] == 'p')
          ? cArrs[i]->permute(modifC[i], false, false)
          : cArrs[i]->reshape(
              c->ordering(), modifC[i],
              false);  // since we ignore first element in cArrs (that is cArrs[0]) then it is always equal to c
  }

  mmul(aPR, bPR, cArrs[cArrs.size() - 1], 1.0, 0.0);

  // check whether new buffer allocation was happened for c array
  if (!whatToDoWithC.empty()) {
    for (int i = cArrs.size() - 1; i > 0; --i) {
      if (cArrs[i]->buffer() != cArrs[i - 1]->buffer() || cArrs[i]->specialBuffer() != cArrs[i - 1]->specialBuffer())
        cArrs[i - 1]->assign(cArrs[i]);
      delete cArrs[i];
    }
  }

  if (aPR != a) delete aPR;
  if (bPR != b) delete bPR;
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* a, NDArray* b,
                               std::vector<std::vector<LongType>>& modifA,
                               std::vector<std::vector<LongType>>& modifB) {
  NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
  std::string whatToDoWithA,
      whatToDoWithB;  // "" - nothing; "p" - permutation only; "r" - reshaping only; "pr" - permutation+reshaping; "rp"
  // - reshaping/permutation; another string - throw exception

  for (const auto& arr : modifA)
    whatToDoWithA =
        (std::find(arr.begin(), arr.end(), 0) != arr.end())
        ? whatToDoWithA + "p"
        : whatToDoWithA +
          "r";  // when 0 is present in arr then it is permutation array, otherwise - it is reshaping array
  for (const auto& arr : modifB)
    whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithB + "p" : whatToDoWithB + "r";

  // first step for a array - use view (no copy) when possible
  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0], false, false)
                                    : a->reshape(a->ordering(), modifA[0], false);
  // first step for b array - use view (no copy) when possible
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0], false, false)
                                    : b->reshape(b->ordering(), modifB[0], false);
  // rest steps for a array
  for (size_t i = 1; i < whatToDoWithA.size(); ++i)
    if (whatToDoWithA[i] == 'p')
      aPR->permutei(modifA[i], false, false);
    else
      aPR->reshapei(modifA[i]);
  // rest steps for b array
  for (size_t i = 1; i < whatToDoWithB.size(); ++i)
    if (whatToDoWithB[i] == 'p')
      bPR->permutei(modifB[i], false, false);
    else
      bPR->reshapei(modifB[i]);

  NDArray* result = mmul(aPR, bPR, nullptr, 1.0, 0.0);

  return result;
}
#endif

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmul(NDArray* A, NDArray* B, NDArray* C, const double alpha,
                          const double beta, const char outOrder) {
  LongType lenDim;
  const LongType aRank = A->rankOf();
  const LongType bRank = B->rankOf();
  const bool isAVector = shape::isCommonVector(A->shapeInfo(), lenDim);
  const bool isBVector = shape::isCommonVector(B->shapeInfo(), lenDim);
  // dot product of 2 vectors
  if (A->lengthOf() == B->lengthOf() && isAVector && isBVector &&
      (aRank != 2 ||
       (aRank == 2 && (A->isSameShape(B) ||
                      (bRank == 1 && A->sizeAt(1) == 1))))) {  // (1x1x1 * 1x1) or (1x4 * 1*4) or (4x1 * 4x1) or (4x1 * 4)


    return dot(A, B, C, alpha, beta);
  }
  // matrix x matrix
  if (aRank == 2 && bRank == 2) {
    return mmulMxM(A, B, C, alpha, beta, outOrder);
  }

  // matrix x vector
  if (aRank == 2 && isBVector) {
    return mmulMxV(A, B, C, alpha, beta, outOrder);
  }

  // vector x matrix, A{M} x B{M,N} = C{N} -> reduce to matrix x matrix A2{1,M} x B{M,N} = C2{1,N}, since there is no
  // corresponding blas operation sgevm
  if (isAVector && bRank == 2) {
    std::vector<sd::LongType> aShape = {1, A->lengthOf()};
    std::vector<sd::LongType> cShape = {1, C->lengthOf()};


    NDArray* A2 = A->reshape(A->ordering(), aShape);                       // A{M} -> A2{1,M}
    NDArray* C2 = C ? C->reshape(C->ordering(), cShape, false) : nullptr;  // C{N} -> C2{1,N}
    auto result = mmulMxM(A2, B, C2, alpha, beta, outOrder);                                        // result{1,N}

    // Cleanup reshaped arrays
    if (A2 != A) delete A2;
    if (C2 != nullptr && C2 != C) delete C2;

    if (!C) {
      result->reshapei({result->lengthOf()});  // result{1,N} -> result{N}
      return result;
    }
    return C;
  }

  // batched matrix multiplication
  return mmulNxN(A, B, C, alpha, beta, outOrder);
}

bool MmulHelper::resolveTranspose(sd::NDArray& a, sd::NDArray& b, bool& transA, bool& transB) {
  int rowsA = a.sizeAt(-2);
  int colsA = a.sizeAt(-1);
  int rowsB = b.sizeAt(-2);
  int colsB = b.sizeAt(-1);

  transA = false;
  transB = false;


  if (colsA == rowsB) {
    // No transpose needed
    return true;
  } else if (rowsA == rowsB) {
    // Transpose A
    transA = true;
    return true;
  } else if (colsA == colsB) {
    // Transpose B
    transB = true;
    return true;
  } else {
    // Dimensions do not match for matrix multiply
    return false;
  }
}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::matmul(NDArray* x, NDArray* y, NDArray* z, const bool transX, const bool transY, double alpha,
                        double beta, NDArray* realFinalResult) {
  int xRank = x->rankOf();
  int yRank = y->rankOf();

  auto outShape = ShapeUtils::evalShapeForMatmul(x->shapeInfo(), y->shapeInfo(), transX, transY);
  if (!z->isSameShape(outShape)) {
    std::string errorMessage;
    errorMessage = "NDArrayFactory::matmul static method: input shape of output array is wrong, actual is";
    errorMessage += ShapeUtils::shapeAsString(z).c_str();
    errorMessage += " and expected is ";
    errorMessage += ShapeUtils::shapeAsString(outShape).c_str();
    errorMessage += " ! \n";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (z->isEmpty()) return;

  NDArray *xT = const_cast<NDArray *>(x);
  NDArray *yT = const_cast<NDArray *>(y);
  NDArray *zT = z;

  // Handle transpose via permute + dup for contiguous data
  // permute creates a view with swapped strides, dup() makes a contiguous copy
  if ((transX && xRank > 1) || (transY && yRank > 1)) {
    const int rank = xRank >= yRank ? xRank : yRank;
    std::vector<LongType> permut(rank);
    for (int i = 0; i < rank - 2; ++i) permut[i] = i;
    permut[rank - 2] = rank - 1;
    permut[rank - 1] = rank - 2;

    if (transX) {
      NDArray *permutedView = x->permute(permut, false, false);  // Create view (non-contiguous)
      xT = permutedView->dup();  // Make contiguous copy with proper data layout
      delete permutedView;
    }
    if (transY) {
      NDArray *permutedView = y->permute(permut, false, false);  // Create view (non-contiguous)
      yT = permutedView->dup();  // Make contiguous copy with proper data layout
      delete permutedView;
    }
  }

  if (xRank <= 2 && yRank <= 2) {
    // dot (1Dx1D), vector-matrix (1Dx2D), matrix-vector (2Dx1D), matrix-matrix (2Dx2D) product cases
    NDArray* xReshaped = nullptr;
    NDArray* zReshaped = nullptr;
    
    if (xRank == 1 && yRank == 2) {
      // reduce vector-matrix to matrix-matrix case
      std::vector<sd::LongType> xShape = {1, xT->lengthOf()};
      std::vector<sd::LongType> zShape = {1, z->lengthOf()};

      // Remember if we need to delete the permuted versions
      NDArray* xPermuted = (xT != x) ? xT : nullptr;
      NDArray* zPermuted = (zT != z) ? zT : nullptr;

      xReshaped = xT->reshape(xT->ordering(), xShape, false);
      xT = xReshaped;
      zReshaped = z->reshape(z->ordering(), zShape, false);
      zT = zReshaped;

      // Clean up permuted versions if they exist
      if(xPermuted != nullptr && !xPermuted->isView()) {
        delete xPermuted;
      }
      if(zPermuted != nullptr && !zPermuted->isView()) {
        delete zPermuted;
      }
    }

    mmul(xT, yT, zT, alpha, beta);

    // Copy back result and clean up reshaped output
    if(zT != z) {
      z->dataBuffer()->copyBufferFrom(*zT->dataBuffer(), zT->lengthOf() * zT->sizeOfT());
      delete zT;
      zT = z;  // Reset to original to prevent double-free at end of function
    }

    // Clean up reshaped input
    if(xReshaped != nullptr && xReshaped != x) {
      delete xReshaped;
      xT = x;  // Reset to original to prevent double-free at end of function
    }

  } else {
    // Batched matmul: loop over batch dimensions and call 2D gemm for each slice
    // This is more reliable than mmulNxN which has bugs in batch index calculation

    // For 3D arrays [batch, M, K] x [batch, K, N] = [batch, M, N]
    // We iterate over batch dimension and call 2D mmul for each slice
    const int xRankT = xT->rankOf();
    const int yRankT = yT->rankOf();
    const int zRankT = zT->rankOf();

    if (xRankT == 3 && yRankT == 3 && zRankT == 3) {
      // Simple case: all 3D with matching batch dimension
      const LongType batchSize = xT->sizeAt(0);
      const int M = static_cast<int>(xT->sizeAt(1));
      const int K = static_cast<int>(xT->sizeAt(2));
      const int N = static_cast<int>(yT->sizeAt(2));

      const auto dtype = xT->dataType();
      const bool hasBatchedFloat = (dtype == DataType::FLOAT32) && BlasHelper::getInstance().hasBatchedGEMM<float>();
      const bool hasBatchedDouble = (dtype == DataType::DOUBLE) && BlasHelper::getInstance().hasBatchedGEMM<double>();

      if ((hasBatchedFloat || hasBatchedDouble) && Environment::getInstance().isEnableBlas()) {
        // Use batched GEMM - process all batches in single BLAS call
        const int batchCount = static_cast<int>(batchSize);

        // Allocate arrays for batch parameters
        std::vector<CBLAS_TRANSPOSE> transA_arr(batchCount, CblasNoTrans);
        std::vector<CBLAS_TRANSPOSE> transB_arr(batchCount, CblasNoTrans);
        std::vector<int> M_arr(batchCount, M);
        std::vector<int> N_arr(batchCount, N);
        std::vector<int> K_arr(batchCount, K);
        std::vector<int> lda_arr(batchCount);
        std::vector<int> ldb_arr(batchCount);
        std::vector<int> ldc_arr(batchCount);

        // Set up pointer arrays
        std::vector<float*> A_arr_f, B_arr_f, C_arr_f;
        std::vector<double*> A_arr_d, B_arr_d, C_arr_d;
        std::vector<float> alpha_arr_f, beta_arr_f;
        std::vector<double> alpha_arr_d, beta_arr_d;

        for (int b = 0; b < batchCount; ++b) {
          // Calculate strides for this batch
          lda_arr[b] = static_cast<int>(xT->strideAt(1));  // stride along M
          ldb_arr[b] = static_cast<int>(yT->strideAt(1));  // stride along K
          ldc_arr[b] = static_cast<int>(zT->strideAt(1));  // stride along M

          if (hasBatchedFloat) {
            A_arr_f.push_back(xT->bufferAsT<float>() + b * xT->strideAt(0));
            B_arr_f.push_back(yT->bufferAsT<float>() + b * yT->strideAt(0));
            C_arr_f.push_back(zT->bufferAsT<float>() + b * zT->strideAt(0));
            alpha_arr_f.push_back(static_cast<float>(alpha));
            beta_arr_f.push_back(static_cast<float>(beta));
          } else {
            A_arr_d.push_back(xT->bufferAsT<double>() + b * xT->strideAt(0));
            B_arr_d.push_back(yT->bufferAsT<double>() + b * yT->strideAt(0));
            C_arr_d.push_back(zT->bufferAsT<double>() + b * zT->strideAt(0));
            alpha_arr_d.push_back(alpha);
            beta_arr_d.push_back(beta);
          }
        }

        int groupSize = batchCount;
        auto blasLock = BlasHelper::getInstance().lockBlas();

        if (hasBatchedFloat) {
          BlasHelper::getInstance().sgemmBatched()(
              CblasRowMajor, transA_arr.data(), transB_arr.data(),
              M_arr.data(), N_arr.data(), K_arr.data(),
              alpha_arr_f.data(), A_arr_f.data(), lda_arr.data(),
              B_arr_f.data(), ldb_arr.data(),
              beta_arr_f.data(), C_arr_f.data(), ldc_arr.data(),
              1, &groupSize);
        } else {
          BlasHelper::getInstance().dgemmBatched()(
              CblasRowMajor, transA_arr.data(), transB_arr.data(),
              M_arr.data(), N_arr.data(), K_arr.data(),
              alpha_arr_d.data(), A_arr_d.data(), lda_arr.data(),
              B_arr_d.data(), ldb_arr.data(),
              beta_arr_d.data(), C_arr_d.data(), ldc_arr.data(),
              1, &groupSize);
        }
      } else {
        // Fallback: serial BLAS loop or parallel element-wise
        const LongType matrixSize = M * K * N;
        if (matrixSize > 50000) {
          for (LongType b = 0; b < batchSize; ++b) {
            auto xSlice = (*xT)(b, {0});
            auto ySlice = (*yT)(b, {0});
            auto zSlice = (*zT)(b, {0});
            mmul(xSlice, ySlice, zSlice, alpha, beta);
            delete xSlice;
            delete ySlice;
            delete zSlice;
          }
        } else {
          mmulNxN(xT, yT, zT, alpha, beta, z->ordering());
        }
      }
    } else {
      // Fall back to mmulNxN for other cases (4D+, mixed ranks, etc.)
      mmulNxN(xT, yT, zT, alpha, beta, z->ordering());
    }
  }

  // Clean up permuted arrays (works for both cases)
  if (xT != x && xT != nullptr) delete xT;
  if (yT != y && yT != nullptr) delete yT;

  if(realFinalResult != nullptr && realFinalResult != z) {
    realFinalResult->dataBuffer()->copyBufferFrom(*z->dataBuffer());
  }


}
}  // namespace sd

#endif
