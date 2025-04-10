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
  NDArray* aP = permutAt.empty() ? A : new NDArray(A->permute(permutAt, false, false));
  NDArray* bP = permutBt.empty() ? B : new NDArray(B->permute(permutBt, false, false));

  // check whether reshape is necessary
  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : new NDArray(aP->reshape(aP->ordering(), shapeAt));
  NDArray* bPR = bP->isSameShape(shapeAt) ? bP : new NDArray(bP->reshape(bP->ordering(), shapeBt));

  NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);

  c->reshapei(outShape);

  return c;
}


void MmulHelper::computeNewShapesAndAxes(
    NDArray& as_, const std::vector<LongType>& axes_a,
    NDArray& bs, const std::vector<LongType>& axes_b,
    std::vector<LongType>& newshape_a, std::vector<LongType>& newaxes_a,
    std::vector<LongType>& newshape_b, std::vector<LongType>& newaxes_b
) {


  std::vector<LongType> as_shape = as_.getShapeAsVector();
  std::vector<LongType> bs_shape = bs.getShapeAsVector();

  std::vector<LongType> notin_a;
  for(size_t k = 0; k < as_shape.size(); ++k) {
    if(std::find(axes_a.begin(), axes_a.end(), k) == axes_a.end())
      notin_a.push_back(k);
  }



  newaxes_a.clear();
  std::copy(notin_a.begin(), notin_a.end(), std::back_inserter(newaxes_a));
  std::copy(axes_a.begin(), axes_a.end(), std::back_inserter(newaxes_a));

  LongType N2_a = std::accumulate(axes_a.begin(), axes_a.end(), 1L, [&](LongType product, LongType i){
    return product * as_shape[i];
  });

  newshape_a.clear();
  newshape_a.push_back(std::accumulate(notin_a.begin(), notin_a.end(), 1L, [&](LongType product, LongType i){
    return product * as_shape[i];
  }));
  newshape_a.push_back(N2_a);



  std::vector<LongType> notin_b;
  for(size_t k = 0; k < bs_shape.size(); ++k) {
    if(std::find(axes_b.begin(), axes_b.end(), k) == axes_b.end())
      notin_b.push_back(k);
  }


  newaxes_b.clear();
  std::copy(axes_b.begin(), axes_b.end(), std::back_inserter(newaxes_b));
  std::copy(notin_b.begin(), notin_b.end(), std::back_inserter(newaxes_b));



  LongType N2_b = std::accumulate(axes_b.begin(), axes_b.end(), 1L, [&](LongType product, LongType i){
    return product * bs_shape[i];
  });



  newshape_b.clear();
  newshape_b.push_back(N2_b);
  newshape_b.push_back(std::accumulate(notin_b.begin(), notin_b.end(), 1L, [&](LongType product, LongType i){
    return product * bs_shape[i];
  }));


}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot2(NDArray* a, NDArray* b, NDArray* c, const std::vector<LongType>& axes_a,
                            const std::vector<LongType>& axes_b, std::vector<LongType>& permutAt,
                            std::vector<LongType>& permuteBt, std::vector<LongType>& permuteCt,
                            NDArray* realFinalResult) {

  // check whether permutation is required
  NDArray* cP  =permuteCt.empty() ? c : new NDArray(c->permute(permuteCt, false, false));

  std::vector<LongType> shapeAt, shapeBt;
  std::vector<LongType> permutAtDummy, permuteBtDummy;

  std::vector<LongType> newshape_a, newaxes_a, newshape_b, newaxes_b;
  computeNewShapesAndAxes(*a, axes_a, *b, axes_b, newshape_a, newaxes_a, newshape_b, newaxes_b);

  NDArray* aP = permutAt.empty() ? a : new NDArray(a->permute(permutAt, false, false));
  NDArray* bP = permuteBt.empty() ? b : new NDArray(b->permute(permuteBt, false, false));

  auto apReshaped = aP->permute(newaxes_a, false, false).reshape('c', newshape_a,true);
  NDArray* aPR =  new NDArray(apReshaped);

  auto bpReshape = bP->permute(newaxes_b, false, false).reshape('c', newshape_b,true);
  NDArray* bPR = new NDArray(bpReshape);

  std::vector<LongType> requiredCshape  = {aPR->sizeAt(0), bPR->sizeAt(1)};
  NDArray cP2 = cP->reshape('f', requiredCshape, false);
  NDArray* cPR = new NDArray(cP2);

  NDArray * ret = mmul(aPR, bPR, cPR, 1.0, 0.0);

  if (cPR->buffer() != cP->buffer() ||
      cPR->specialBuffer() != cP->specialBuffer()) {  // this means both permute and reshape have been performed on c, cP
    if(c->buffer() == cP->buffer()) {
      auto copyFromBuff = cP->dataBuffer();
      cP->dataBuffer()->copyBufferFrom(*copyFromBuff);
    } else {
      auto copyFromBuff = cP->dataBuffer();
      c->dataBuffer()->copyBufferFrom(*copyFromBuff);
    }
  }

  if(realFinalResult != c) {
    realFinalResult->dataBuffer()->copyBufferFrom(*c->dataBuffer());
  }

}


void MmulHelper::tensorDot(NDArray* a, NDArray* b, NDArray* c,
                           std::vector<LongType>& axes_a, std::vector<LongType>& axes_b,
                           std::vector<LongType>& permutForC) {

  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;
  ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);


  // check whether permutation is required
  NDArray* cP = permutForC.empty() ? c : new NDArray(c->permute(permutForC, false, false));
  // check whether permutation is necessary
  NDArray* aP = permutAt.empty() ? a : new NDArray(a->permute(permutAt, false, false));
  NDArray* bP = permutBt.empty() ? b : new NDArray(b->permute(permutBt, false, false));

  // check whether reshape is necessary
  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : new NDArray(aP->reshape(aP->ordering(), shapeAt));
  NDArray* bPR = bP->isSameShape(shapeAt) ? bP : new NDArray(bP->reshape(bP->ordering(), shapeBt));

  std::vector<LongType> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};


  NDArray* cPR = cP->isSameShape(requiredCshape) ? cP : new NDArray(cP->reshape(cP->ordering(), requiredCshape, false));
  NDArray *ret = mmul(aPR, bPR, cPR, 1.0, 0.0);

  if (c != ret) {  // this means both permute and reshape have been performed on c, cP
    // always points on c->buffer()
    NDArray assign2 = ret->reshape(c->ordering(),requiredCshape);
    c->assign(&assign2);
  }


  if(c != cP) {
    delete cP;
  }

  if(aP != a) {
    delete aP;
  }

  if(bP != b) {
    delete bP;
  }

  if(aPR != a) {
    delete aPR;
  }
  if(bPR != b) {
    delete bPR;
  }

  if(cPR != c) {
    delete cPR;
  }
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

  // first step for a array

  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? new NDArray(a->permute(modifA[0], false, false))
                                    : new NDArray(a->reshape(a->ordering(), modifA[0]));
  // first step for b array
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? new NDArray(b->permute(modifB[0], false, false))
                                    : new NDArray(b->reshape(b->ordering(), modifB[0]));
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
          ? new NDArray(cArrs[i]->permute(modifC[i], false, false))
          : new NDArray(cArrs[i]->reshape(
              c->ordering(), modifC[i],
              false));  // since we ignore first element in cArrs (that is cArrs[0]) then it is always equal to c
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

  // first step for a array
  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? new NDArray(a->permute(modifA[0], false, false))
                                    : new NDArray(a->reshape(a->ordering(), modifA[0]));
  // first step for b array
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? new NDArray(b->permute(modifB[0], false, false))
                                    : new NDArray(b->reshape(b->ordering(), modifB[0]));
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


    NDArray* A2 = new NDArray(A->reshape(A->ordering(),aShape));                       // A{M} -> A2{1,M}
    NDArray* C2 = C ? new NDArray(C->reshape(C->ordering(), cShape, false)) : nullptr;  // C{N} -> C2{1,N}
    auto result = mmulMxM(A2, B, C2, alpha, beta, outOrder);                                        // result{1,N}


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

  if ((transX && xRank > 1) || (transY && yRank > 1)) {
    const int rank = xRank >= yRank ? xRank : yRank;
    std::vector<LongType> permute(rank);
    for (int i = 0; i < rank - 2; ++i) permute[i] = i;
    permute[rank - 2] = rank - 1;
    permute[rank - 1] = rank - 2;

    //transpose can affect the input data. We shouldn't mutate that.
    //note we dup here to avoid manipulating the reference
    if (transX) {
      NDArray &permuted = x->permute(permute, true, false);
      xT = new NDArray(permuted);
    }
    if (transY) {
      NDArray &yPermuted = y->permute(permute, true, false);
      yT = new NDArray(yPermuted);

    }
  }

  if (xRank <= 2 && yRank <= 2) {
    // dot (1Dx1D), vector-matrix (1Dx2D), matrix-vector (2Dx1D), matrix-matrix (2Dx2D) product cases
    if (xRank == 1 && yRank == 2) {
      // reduce vector-matrix to matrix-matrix case
      //note we dup to avoid mutating input data
      std::vector<sd::LongType> xShape = {1, xT->lengthOf()};
      std::vector<sd::LongType> zShape = {1, z->lengthOf()};
      NDArray &xReshape = x->dup(x->ordering()).reshape(xT->ordering(), xShape,false);
      xT = new NDArray(xReshape);  // please note x is not transposed in this case (since xRank=1)
      NDArray &zReshape = z->dup(z->ordering()).reshape(z->ordering(), zShape,false);
      zT = new NDArray(zReshape);
    }


    mmul(xT, yT, zT, alpha, beta);


    if(zT != z) {
      z->dataBuffer()->copyBufferFrom(*zT->dataBuffer(), zT->lengthOf() * zT->sizeOfT());
    }


  } else {
    // rest cases - batched mmul
    const int batchRank = xRank - 2;
    std::vector<LongType> dimsToExclude;
    for (int i = 0; i < batchRank; ++i) {
      dimsToExclude.push_back(i);
    }

    const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(xT->shapeInfo(), dimsToExclude);

    std::vector<NDArray*> vA(numOfSubArrs);
    std::vector<NDArray*> vB(numOfSubArrs);
    std::vector<NDArray*> vC(numOfSubArrs);

    for (LongType i = 0; i < numOfSubArrs; ++i) {
      vA[i] = new NDArray((*xT)(i, dimsToExclude));
      vB[i] = new NDArray((*yT)(i, dimsToExclude));
      vC[i] = new NDArray((*zT)(i, dimsToExclude));
    }

    NDArray alphaArr = NDArrayFactory::create<double>('c', {0}, {alpha});
    NDArray betaArr = NDArrayFactory::create<double>('c', {0}, {beta});

    int M = vA[0]->sizeAt(0);
    int N = vB[0]->sizeAt(1);
    int K = vA[0]->sizeAt(1);
    int lda = vA[0]->sizeAt(0);
    int ldb = vB[0]->sizeAt(0);
    int ldc = vC[0]->sizeAt(0);

    bool transXResolve = transX == 1;
    bool transYResolve = transY == 1;
    if(!resolveTranspose(*vA[0], *vB[0], transXResolve, transYResolve)) {
      // Batch dimensions do not match
      std::string errorMessage;
      errorMessage = "NDArrayFactory::matmul static method: batch dimensions do not match";
      errorMessage += "x shape: ";
      errorMessage += ShapeUtils::shapeAsString(vA[0]).c_str();
      errorMessage += " y shape: ";
      errorMessage += ShapeUtils::shapeAsString(vB[0]).c_str();
      errorMessage += " ! \n";
      errorMessage += "z shape: ";
      errorMessage += ShapeUtils::shapeAsString(vC[0]).c_str();
      THROW_EXCEPTION(errorMessage.c_str());

    }

    ops::helpers::bgemm(vA, vB, vC, &alphaArr, &betaArr, transXResolve ? 1 : 0, transYResolve ? 1 : 0, M, N, K, lda, ldb, ldc);

    for (LongType i = 0; i < numOfSubArrs; ++i) {
      delete vA[i];
      delete vB[i];
      delete vC[i];
    }


  }

  if (xT != x) delete xT;
  if (yT != y) delete yT;


  if(realFinalResult != nullptr && realFinalResult != z) {
    realFinalResult->dataBuffer()->copyBufferFrom(*z->dataBuffer());
  }

}
}  // namespace sd

#endif
