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

CUSTOM_OP_IMPL(batched_gemm, -1, -1, false, 0, 9) {
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);
  if(transA == 0) {
    int ldaComp = M > 1 ? M : 1;
    if(ldA < ldaComp) throw std::runtime_error("LDA must be >= max(1,m) when transa == false");
  } else {
    int ldaComp = K > 1 ? K : 1;
    if(ldA < ldaComp)
      throw std::runtime_error("LDA must be >= max(1,k) when transa == true");
  }

  if(transB == 0) {
    int ldBComp = K > 1 ? K : 1;
    if(ldB < ldBComp) {
      throw std::runtime_error("LDB must be >= max(1,k) when transb == false");
    }
  } else {
    int ldbComp = N > 1 ? N : 1;
    if(ldB < ldbComp)
      throw std::runtime_error("LDB must be >= max(1,N) when transb == true");
  }

  int ldcComp = M > 1 ? M : 1;
  if(ldC < ldcComp) {
    throw std::runtime_error("LDC must be < max(1,M) when transc != false");
  }


  if (transA == 0) transA = 111;

  if (transB == 0) transB = 111;

  if (transA == 1) transA = 112;

  if (transB == 1) transB = 112;
  if(M < 0) throw std::runtime_error("M < 0");
  if(N < 0) throw std::runtime_error("N < 0");
  if(K < 0) throw std::runtime_error("K < 0");

  REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0,
               "BatchedGemm: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
  REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0, 0, "");


  auto alpha = INPUT_VARIABLE(0);
  NDArray *alphaInput = nullptr;
  if(alpha->lengthOf() != batchSize) {
    alphaInput = new NDArray('c',{batchSize},alpha->dataType());
    alphaInput->assign(alpha);
  } else {
    alphaInput = alpha;
  }


  auto beta = INPUT_VARIABLE(1);
  NDArray *betaInput = nullptr;
  if(beta->lengthOf() != batchSize) {
    betaInput = new NDArray('c',{batchSize},beta->dataType());
    betaInput->assign(beta);
  } else {
    betaInput = beta;
  }

  sd_printf("After alphs and betas\n",0);
  std::vector<NDArray*> vA(batchSize);
  std::vector<NDArray*> vB(batchSize);
  std::vector<NDArray*> vC(batchSize);

  auto firstType = INPUT_VARIABLE(0)->dataType();
  for (int e = 0; e < batchSize; e++) {
    vA[e] = INPUT_VARIABLE(e + 2);
    vB[e] = INPUT_VARIABLE(e + 2 + batchSize);
    vC[e] = OUTPUT_VARIABLE(e);

    REQUIRE_TRUE(firstType == vC[e]->dataType(), 0, "BatchedGemm: all inputs and outputs must have same data type");

    REQUIRE_TRUE(vA[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of A should be equal to 2", e);
    REQUIRE_TRUE(vB[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of B should be equal to 2", e);
    REQUIRE_TRUE(vC[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of C should be equal to 2", e);

    if(transA == 111) {
      REQUIRE_TRUE(M == vA[e]->sizeAt(0), 0, "BatchedGemm: batch %i, number of A.rows() should be equal to M transA: false", e);
      REQUIRE_TRUE(K == vA[e]->sizeAt(1) , 0,
                   "BatchedGemm: batch %i, number of A.columns() should be equal to K transA: false", e);
    } else  {
      REQUIRE_TRUE(M == vA[e]->sizeAt(1), 0, "BatchedGemm: batch %i, number of A.columns() should be equal to M transA: true", e);
      REQUIRE_TRUE(K == vA[e]->sizeAt(0) , 0,
                   "BatchedGemm: batch %i, number of A.rows() should be equal to K transA: true", e);
    }

    if(transB == 111) {
      REQUIRE_TRUE(N == vB[e]->sizeAt(1), 0, "BatchedGemm: batch %i, number of B.rows() should be equal to N transB: false", e);
      REQUIRE_TRUE(K == vA[e]->sizeAt(0) , 0,
                   "BatchedGemm: batch %i, number of B.rows() should be equal to K transB: false", e);
    } else {
      REQUIRE_TRUE(N == vB[e]->sizeAt(0), 0, "BatchedGemm: batch %i, number of B.columns() should be equal to N transB: true", e);
      REQUIRE_TRUE(K == vA[e]->sizeAt(1) , 0,
                   "BatchedGemm: batch %i, number of B.rows() should be equal to K transB: true", e);
    }



  }

  REQUIRE_TRUE(vA.size() == vB.size() && vA.size() == vC.size() && vA.size() == batchSize, 0,
               "BatchedGemm: mismatched numbers of A, B, C for unknown reason");

  sd_printf("Before bgemm execution\n",0);
  sd::ops::helpers::bgemm(vA,
                          vB,
                          vC,
                          alphaInput,
                          betaInput,
                          transA,
                          transB,
                          M,
                          N,
                          K,
                          ldA,
                          ldB,
                          ldC);
  sd_printf("After bgemm execution\n",0);




  return sd::Status::OK;
};

DECLARE_SHAPE_FN(batched_gemm) {
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);

  auto firstType = ArrayOptions::dataType(inputShape->at(0));
  for (int e = 1; e < block.width(); e++) {
    REQUIRE_TRUE(firstType == ArrayOptions::dataType(inputShape->at(1)), 0,
                 "BatchedGemm: all inputs must have same data type");
  }

  auto shapeList = SHAPELIST();

  if (!(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0)) {
    shapeList->push_back(
        ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShape->at(0)), 'c', {1, 1}));
    return shapeList;
  }

  std::vector<sd::LongType> shape({M, N});

  for (int e = 0; e < batchSize; e++) {
    auto newShape =
        ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShape->at(0)), 'f', shape);
    shapeList->push_back(newShape);
  }

  return shapeList;
}

DECLARE_TYPES(batched_gemm) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}



CUSTOM_OP_IMPL(batched_gemm_bp, -1, -1, false, 0, 9) {
  sd_printf("Block input size: %d Output width: %d \n",block.width(),block.outputWidth());


  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);
  if(transA == 0) {
    int ldaComp = M > 1 ? M : 1;
    if(ldA < ldaComp) throw std::runtime_error("LDA must be >= max(1,m) when transa == false");
  } else {
    int ldaComp = K > 1 ? K : 1;
    if(ldA < ldaComp)
      throw std::runtime_error("LDA must be >= max(1,k) when transa == true");
  }

  if(transB == 0) {
    int ldBComp = K > 1 ? K : 1;
    if(ldB < ldBComp) {
      throw std::runtime_error("LDB must be >= max(1,k) when transb == false");
    }
  } else {
    int ldbComp = N > 1 ? N : 1;
    if(ldB < ldbComp)
      throw std::runtime_error("LDB must be >= max(1,N) when transb == true");
  }

  int ldcComp = M > 1 ? M : 1;
  if(ldC < ldcComp) {
    throw std::runtime_error("LDC must be < max(1,M) when transc != false");
  }


  if (transA == 0) transA = 111;

  if (transB == 0) transB = 111;

  if (transA == 1) transA = 112;

  if (transB == 1) transB = 112;
  if(M < 0) throw std::runtime_error("M < 0");
  if(N < 0) throw std::runtime_error("N < 0");
  if(K < 0) throw std::runtime_error("K < 0");

  REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0,
               "BatchedGemm: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
  REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0, 0, "");

  batched_gemm batchedGemm;

  std::vector<NDArray *> matricesA;
  std::vector<NDArray *> matricesB;
  std::vector<NDArray *> dlDOut;
  std::vector<NDArray *> dldXOutputs;
  std::vector<NDArray *> dldYOutputs;
  sd_printf("Block input size: %d Output width: %d\n",block.width(),block.outputWidth());

  for (int e = 0; e < batchSize; e++) {
    matricesA.push_back(INPUT_VARIABLE(e + 2));
    matricesB.push_back(INPUT_VARIABLE(e + 2 + batchSize));
    dlDOut.push_back(INPUT_VARIABLE(e + 2 + batchSize * 2));
    dldXOutputs.push_back(OUTPUT_VARIABLE(e));
    dldYOutputs.push_back(OUTPUT_VARIABLE(e + batchSize));
  }

  sd_printf("After batch gemm in batch bp\n",0);

  auto alpha = INPUT_VARIABLE(0);
  NDArray *alphaInput = nullptr;
  if(alpha->lengthOf() != batchSize) {
    alphaInput = new NDArray('c',{batchSize},alpha->dataType());
    alphaInput->assign(alpha);
  } else {
    alphaInput = alpha;
  }


  auto beta = INPUT_VARIABLE(1);
  NDArray *betaInput = nullptr;
  if(beta->lengthOf() != batchSize) {
    betaInput = new NDArray('c',{batchSize},beta->dataType());
    betaInput->assign(beta);
  } else {
    betaInput = beta;
  }

  sd::ops::helpers::bgemm(dlDOut, matricesB, dldXOutputs, alphaInput, betaInput, transA, transB, M, N, K, ldA, ldB, ldC);
  sd::ops::helpers::bgemm(matricesA, dlDOut, dldYOutputs, alphaInput, betaInput, transA, transB, M, N, K, ldA, ldB, ldC);


  if(alphaInput != alpha) {
    delete alphaInput;
  }

  if(betaInput != beta) {
    delete betaInput;
  }


  return sd::Status::OK;
};



DECLARE_SHAPE_FN(batched_gemm_bp) {
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);
  sd_printf("Setting DLDX and DLDY shape\n",0);

  batched_gemm batchedGemm;
  std::vector<sd::NDArray *> aInputs;
  std::vector<sd::NDArray *> bInputs;
  std::vector<sd::NDArray *> epsInputs;
  std::vector<sd::NDArray *> alphaBeta;

  auto dldxVarSpace = new VariableSpace();
  auto dldyVarSpace = new VariableSpace();
  auto dldxInputList = SHAPELIST();
  auto dldYInputList = SHAPELIST();
  auto epsVars = std::vector<sd::NDArray *>();
  for(int i = 0; i < 2; i++) {
    alphaBeta.push_back(INPUT_VARIABLE(i));
  }

  sd_printf("Done setting alphas and betas\n",0);

  for(int i = 0; i < batchSize; i++) {
    aInputs.push_back(INPUT_VARIABLE(i + 2));
    bInputs.push_back(INPUT_VARIABLE(batchSize + i + 2));
    epsVars.push_back(INPUT_VARIABLE(i +  2 + (batchSize * 2)));
  }


  aInputs[0]->printShapeInfo("Alpha inputs");
  bInputs[0]->printShapeInfo("Beta inputs");
  epsVars[0]->printShapeInfo("Eps inputs");
  dldxVarSpace->putVariable(0,alphaBeta[0]);
  dldxVarSpace->putVariable(1,alphaBeta[1]);
  dldyVarSpace->putVariable(0,alphaBeta[0]);
  dldyVarSpace->putVariable(1,alphaBeta[1]);

  // 1 x 3, 3 x 4
  for(int i = 0; i < batchSize; i++) {
    dldxVarSpace->putVariable(i + 2,epsVars[i]);
    dldyVarSpace->putVariable(i + 2,aInputs[i]);
  }

  //

  for(int i = 0; i < batchSize; i++) {
    dldxVarSpace->putVariable(i * batchSize + 2,bInputs[i]);
    dldyVarSpace->putVariable(i * batchSize + 2,epsVars[i]);
  }


  for(int i = 0; i < dldxVarSpace->getVariables().size(); i++) {
    dldxInputList->push_back(dldxVarSpace->getVariables()[i]->getNDArray()->shapeInfo());
    dldYInputList->push_back(dldyVarSpace->getVariables()[i]->getNDArray()->shapeInfo());
  }

  sd_printf("A inputs size %d\n",aInputs.size());
  sd_printf("B inputs size %d\n",bInputs.size());

  sd_printf("After batch size \n",0);



  auto ctx = graph::Context(0, dldxVarSpace);

  auto dldxAShape = dldxVarSpace->getVariables()[2]->getNDArray();
  auto dldxBShape = dldxVarSpace->getVariables()[2 + batchSize]->getNDArray();
  for(int i = 0; i < dldxVarSpace->getVariables().size(); i++) {
    sd_printf("DLDX var space shape info at %d\n",i);
    dldxVarSpace->getVariables()[i]->getNDArray()->printShapeInfo("DLDX shape info:");
  }

  int transa1 = 0;
  int transb1 = 0;
  int M1 = transa1 > 0 ? (int) aInputs[0]->sizeAt(1) : (int)  aInputs[0]->sizeAt(0);
  int N1 = transb1 > 0? (int) bInputs[0]->sizeAt(0): (int)dldxBShape->sizeAt(1);
  int K1 = transb1 > 0 ? (int) aInputs[0]->sizeAt(0) : (int) aInputs[0]->sizeAt(1) ;
  int lda1 = (int)  aInputs[0]->sizeAt(0);
  int ldb1 = (int) bInputs[0]->sizeAt(0);
  int ldc1 = (int) aInputs[0]->sizeAt(0);

  ctx.setIArguments({transa1,transb1,M1,N1,K1,lda1,ldb1,ldc1,batchSize});
  dldxAShape->printShapeInfo("DLDX shape:");
  dldxBShape->printShapeInfo("DLDY Shape:");
  sd_printf("M1: %d N1: %d K1: %d lda1: %d ldb1 %d ldc %d\n",M1,N1,K1,lda1,ldb1,ldc1);
  //eps: 1 x 4 4 x 3: 1 x 3
  auto dldxShape = batchedGemm.calculateOutputShape(dldxInputList,ctx);
  sd_printf("DLDX shape size: %d\n",dldxShape->size());
  shape::printShapeInfo(dldxShape->at(0));



  auto bCtx = graph::Context(0, dldyVarSpace);

  auto dldyAShape = dldyVarSpace->getVariables()[2];
  auto dldyBShape = dldyVarSpace->getVariables()[2 + batchSize];
  int transa2 = transA;
  int transb2 = 0;
  int M2 = transa2 > 0 ? (int) dldyAShape->getNDArray()->sizeAt(0): (int)  dldyAShape->getNDArray()->sizeAt(1);
  int N2 = transb2 > 0? (int)  dldyBShape->getNDArray()->sizeAt(1): (int) dldyBShape->getNDArray()->sizeAt(0);
  int K2 = transb2 > 0 ? (int) dldyBShape->getNDArray()->sizeAt(1): (int) dldyBShape->getNDArray()->sizeAt(0);
  int lda2 = (int)  dldyAShape->getNDArray()->sizeAt(0);
  int ldb2 = (int) dldyBShape->getNDArray()->sizeAt(0);
  int ldc2 = (int)  dldyAShape->getNDArray()->sizeAt(0);

  bCtx.setIArguments({transa2,transb2,M2,N2,K2,lda2,ldb2,ldc2,batchSize});
  sd_printf("Set i arguments\n",0);
  //TODO: reset a and b list for each batch gemm to be eps slices this may need to be true for op execution as well


  //3 x 4, 1 x 3
  auto dldyShape = batchedGemm.calculateOutputShape(dldYInputList,bCtx);
  shape::printShapeInfo(dldyShape->at(0));

  sd_printf("Set DLDX and DLDY shape\n",0);
  return SHAPELIST(dldxShape->at(0),dldyShape->at(0));
}


DECLARE_TYPES(batched_gemm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}


}  // namespace ops
}  // namespace sd

#endif
