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
//  @author raver119@gmail.com
//   @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_batched_gemm)

#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/batched_gemm.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(batched_gemm, -1, -1, false, 0, 2) {
  // Only require 2 IArgs: transposeA, transposeB
  // Everything else will be inferred from the input matrices
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  
  // Get alpha and beta
  auto alpha = INPUT_VARIABLE(0);
  auto beta = INPUT_VARIABLE(1);
  
  // Calculate batch size from number of inputs
  // Total inputs = alpha + beta + batchSize*A + batchSize*B
  // So batchSize = (total - 2) / 2
  int batchSize = (block.width() - 2) / 2;
  
  REQUIRE_TRUE(batchSize > 0, 0, "BatchedGemm: Invalid batch size calculated: %d", batchSize);
  REQUIRE_TRUE((block.width() - 2) % 2 == 0, 0, "BatchedGemm: Number of matrix inputs must be even");
  
  // Get first matrices to infer dimensions
  auto firstA = INPUT_VARIABLE(2);
  auto firstB = INPUT_VARIABLE(2 + batchSize);
  
  REQUIRE_TRUE(firstA->rankOf() == 2, 0, "BatchedGemm: A matrices must be rank 2");
  REQUIRE_TRUE(firstB->rankOf() == 2, 0, "BatchedGemm: B matrices must be rank 2");
  
  // Infer dimensions from first matrices
  int M = transA ? firstA->sizeAt(1) : firstA->sizeAt(0);
  int K = transA ? firstA->sizeAt(0) : firstA->sizeAt(1);
  int N = transB ? firstB->sizeAt(0) : firstB->sizeAt(1);
  int K_B = transB ? firstB->sizeAt(1) : firstB->sizeAt(0);
  
  REQUIRE_TRUE(K == K_B, 0, "BatchedGemm: Incompatible dimensions - K from A is %d, K from B is %d", K, K_B);
  
  // Infer leading dimensions
  int ldA = firstA->sizeAt(0);
  int ldB = firstB->sizeAt(0);
  int ldC = M;
  
  // Validate leading dimensions
  if(transA == 0) {
    int ldaComp = M > 1 ? M : 1;
    if(ldA < ldaComp) THROW_EXCEPTION("LDA must be >= max(1,m) when transa == false");
  } else {
    int ldaComp = K > 1 ? K : 1;
    if(ldA < ldaComp)
      THROW_EXCEPTION("LDA must be >= max(1,k) when transa == true");
  }

  if(transB == 0) {
    int ldBComp = K > 1 ? K : 1;
    if(ldB < ldBComp) {
      THROW_EXCEPTION("LDB must be >= max(1,k) when transb == false");
    }
  } else {
    int ldbComp = N > 1 ? N : 1;
    if(ldB < ldbComp)
      THROW_EXCEPTION("LDB must be >= max(1,N) when transb == true");
  }

  int ldcComp = M > 1 ? M : 1;
  if(ldC < ldcComp) {
    THROW_EXCEPTION("LDC must be >= max(1,M)");
  }

  // Convert transpose flags to BLAS format
  if (transA == 0) transA = 111;
  if (transB == 0) transB = 111;
  if (transA == 1) transA = 112;
  if (transB == 1) transB = 112;

  REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0,
               "BatchedGemm: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
  REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0, 0, 
               "BatchedGemm: Invalid dimensions M=%d, N=%d, K=%d, ldA=%d, ldB=%d, ldC=%d", M, N, K, ldA, ldB, ldC);

  // Handle alpha and beta
  NDArray *alphaInput = nullptr;
  if(alpha->isScalar()) {
    std::vector<sd::LongType> shape = {batchSize};
    alphaInput = new NDArray('c',shape,alpha->dataType());
    alphaInput->assign(alpha);
  } else {
    alphaInput = alpha;
  }

  NDArray *betaInput = nullptr;
  if(beta->isScalar()) {
    std::vector<LongType> shape = {batchSize};
    betaInput = new NDArray('c',shape,beta->dataType());
    betaInput->assign(beta);
  } else {
    betaInput = beta;
  }

  std::vector<NDArray*> vA(batchSize);
  std::vector<NDArray*> vB(batchSize);
  std::vector<NDArray*> vC(batchSize);

  // Check data types - matrices should all match, alpha/beta can be different
  auto firstMatrixType = firstA->dataType();
  for (int e = 0; e < batchSize; e++) {
    vA[e] = INPUT_VARIABLE(e + 2);
    vB[e] = INPUT_VARIABLE(e + 2 + batchSize);
    vC[e] = OUTPUT_VARIABLE(e);

    REQUIRE_TRUE(firstMatrixType == vA[e]->dataType(), 0, 
                 "BatchedGemm: all A matrices must have same data type");
    REQUIRE_TRUE(firstMatrixType == vB[e]->dataType(), 0, 
                 "BatchedGemm: all B matrices must have same data type");
    REQUIRE_TRUE(firstMatrixType == vC[e]->dataType(), 0, 
                 "BatchedGemm: all output matrices must have same data type as input matrices");
    
    REQUIRE_TRUE(vA[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of A should be equal to 2", e);
    REQUIRE_TRUE(vB[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of B should be equal to 2", e);
    REQUIRE_TRUE(vC[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of C should be equal to 2", e);

    // Verify dimensions are consistent across batch
    int currM = transA == 111 ? vA[e]->sizeAt(0) : vA[e]->sizeAt(1);
    int currK_A = transA == 111 ? vA[e]->sizeAt(1) : vA[e]->sizeAt(0);
    int currK_B = transB == 111 ? vB[e]->sizeAt(0) : vB[e]->sizeAt(1);
    int currN = transB == 111 ? vB[e]->sizeAt(1) : vB[e]->sizeAt(0);
    
    REQUIRE_TRUE(currM == M, 0, "BatchedGemm: batch %i, inconsistent M dimension: expected %d, got %d", e, M, currM);
    REQUIRE_TRUE(currK_A == K, 0, "BatchedGemm: batch %i, inconsistent K dimension in A: expected %d, got %d", e, K, currK_A);
    REQUIRE_TRUE(currK_B == K, 0, "BatchedGemm: batch %i, inconsistent K dimension in B: expected %d, got %d", e, K, currK_B);
    REQUIRE_TRUE(currN == N, 0, "BatchedGemm: batch %i, inconsistent N dimension: expected %d, got %d", e, N, currN);
  }

  helpers::bgemm(vA, vB, vC, alphaInput, betaInput, transA, transB, M, N, K, ldA, ldB, ldC);

  if(alphaInput != alpha) {
    delete alphaInput;
  }

  if(betaInput != beta) {
    delete betaInput;
  }

  return Status::OK;
};

DECLARE_SHAPE_FN(batched_gemm) {
  // Only require 2 IArgs: transposeA, transposeB
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  
  // Calculate batch size from inputs
  int batchSize = (block.width() - 2) / 2;
  
  if (batchSize <= 0) {
    auto shapeList = SHAPELIST();
    shapeList->push_back(
        ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShape->at(0)), 'c', {1, 1}));
    return shapeList;
  }
  
  // Get dimensions from first matrices
  auto firstA = inputShape->at(2);
  auto firstB = inputShape->at(2 + batchSize);
  
  int M = transA ? shape::sizeAt(firstA, 1) : shape::sizeAt(firstA, 0);
  int N = transB ? shape::sizeAt(firstB, 0) : shape::sizeAt(firstB, 1);
  
  // Get data type from first matrix, not from alpha/beta
  auto firstMatrixType = ArrayOptions::dataType(firstA);
  
  // Check that all matrices have the same type (skip alpha and beta)
  for (int e = 2; e < block.width(); e++) {
    REQUIRE_TRUE(firstMatrixType == ArrayOptions::dataType(inputShape->at(e)), 0,
                 "BatchedGemm: all matrices must have same data type");
  }

  auto shapeList = SHAPELIST();
  std::vector<LongType> shape({M, N});

  for (int e = 0; e < batchSize; e++) {
    auto newShape =
        ConstantShapeHelper::getInstance().createShapeInfo(firstMatrixType, 'f', shape);
    shapeList->push_back(newShape);
  }

  return shapeList;
}

DECLARE_TYPES(batched_gemm) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}



CUSTOM_OP_IMPL(batched_gemm_bp, -1, -1, false, 0, 2) {
  // Only require 2 IArgs: transposeA, transposeB
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  
  // Calculate batch size
  // Inputs: alpha, beta, batchSize*A, batchSize*B, batchSize*dLdC
  // So batchSize = (total - 2) / 3
  int batchSize = (block.width() - 2) / 3;
  
  REQUIRE_TRUE(batchSize > 0, 0, "BatchedGemmBp: Invalid batch size calculated: %d", batchSize);
  
  // Get dimensions from first matrices
  auto firstA = INPUT_VARIABLE(2);
  auto firstB = INPUT_VARIABLE(2 + batchSize);
  auto firstDlDOut = INPUT_VARIABLE(2 + batchSize * 2);
  
  int M = transA ? firstA->sizeAt(1) : firstA->sizeAt(0);
  int K = transA ? firstA->sizeAt(0) : firstA->sizeAt(1);
  int N = transB ? firstB->sizeAt(0) : firstB->sizeAt(1);
  
  // Infer leading dimensions
  int ldA = firstA->sizeAt(0);
  int ldB = firstB->sizeAt(0);
  int ldC = M;

  std::vector<NDArray *> matricesA;
  std::vector<NDArray *> matricesB;
  std::vector<NDArray *> dlDOut;
  std::vector<NDArray *> dldXOutputs;
  std::vector<NDArray *> dldYOutputs;

  for (int e = 0; e < batchSize; e++) {
    matricesA.push_back(INPUT_VARIABLE(e + 2));
    matricesB.push_back(INPUT_VARIABLE(e + 2 + batchSize));
    dlDOut.push_back(INPUT_VARIABLE(e + 2 + batchSize * 2));
    //alphas and betas are also set for outputs even though they're zero,every input needs a gradient
    dldXOutputs.push_back(OUTPUT_VARIABLE(e + 2));
    dldYOutputs.push_back(OUTPUT_VARIABLE(e + 2 + batchSize));
  }

  auto alpha = INPUT_VARIABLE(0);
  NDArray *alphaInput = nullptr;
  if(alpha->lengthOf() != batchSize) {
    std::vector<sd::LongType> shape = {batchSize};
    alphaInput = new NDArray('c',shape,alpha->dataType());
    alphaInput->assign(alpha);
  } else {
    alphaInput = alpha;
  }

  auto beta = INPUT_VARIABLE(1);
  NDArray *betaInput = nullptr;
  if(beta->lengthOf() != batchSize) {
    std::vector<sd::LongType> shape = {batchSize};
    betaInput = new NDArray('c',shape,beta->dataType());
    betaInput->assign(beta);
  } else {
    betaInput = beta;
  }

  // Convert transpose flags to BLAS format for helper function
  int transABlas = transA ? 112 : 111;
  int transBBlas = transB ? 112 : 111;

  // First gradient computation: dL/dA = dL/dC @ B^T (or B if transB)
  int transA1 = 0;
  int transB1 = transB;
  int M1 = dlDOut[0]->sizeAt(0);
  int N1 = transB ? matricesB[0]->sizeAt(0) : matricesB[0]->sizeAt(1);
  int k1 = dlDOut[0]->sizeAt(1);
  int lda1 = dlDOut[0]->sizeAt(0);
  int ldb1 = matricesB[0]->sizeAt(0);
  int ldc1 = dldXOutputs[0]->sizeAt(0);
  
  helpers::bgemm(dlDOut, matricesB, dldXOutputs, alphaInput, betaInput, 
                 transA1 ? 112 : 111, transB1 ? 112 : 111, M1, N1, k1, lda1, ldb1, ldc1);

  // Second gradient computation: dL/dB = A^T @ dL/dC (or A if transA)
  int transA2 = transA ? 0 : 1;
  int transB2 = 0;
  int M2 = transA ? matricesA[0]->sizeAt(1) : matricesA[0]->sizeAt(0);
  int N2 = dlDOut[0]->sizeAt(1);
  int k2 = transA ? matricesA[0]->sizeAt(0) : matricesA[0]->sizeAt(1);
  int lda2 = matricesA[0]->sizeAt(0);
  int ldb2 = dlDOut[0]->sizeAt(0);
  int ldc2 = dldYOutputs[0]->sizeAt(0);
  
  helpers::bgemm(matricesA, dlDOut, dldYOutputs, alphaInput, betaInput, 
                 transA2 ? 112 : 111, transB2 ? 112 : 111, M2, N2, k2, lda2, ldb2, ldc2);

  if(alphaInput != alpha) {
    delete alphaInput;
  }

  if(betaInput != beta) {
    delete betaInput;
  }

  return Status::OK;
};

DECLARE_SHAPE_FN(batched_gemm_bp) {
  // Calculate batch size
  int batchSize = (block.width() - 2) / 3;
  
  auto xConstant = CONSTANT(inputShape->at(2));
  auto yConstant = CONSTANT(inputShape->at(2 + batchSize));
  auto ret = SHAPELIST();
  //alpha
  ret->push_back(xConstant);
  //beta
  ret->push_back(yConstant);
  for(int i = 0; i < batchSize; i++) {
    ret->push_back(xConstant);
  }

  for(int i = 0; i < batchSize; i++) {
    ret->push_back(yConstant);
  }
  return ret;
}

DECLARE_TYPES(batched_gemm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif