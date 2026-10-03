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
//
#include <execution/Threads.h>
#include <helpers/BlasHelper.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <ops/op_types.h>
#include <system/env_functions.h>
#include <system/op_boilerplate.h>
#include <types/float16.h>
#include <indexing/NDIndexUtils.h>
#include <ops/declarable/CustomOperations.h>

#if NOT_EXCLUDED(OP_batched_gemm)
namespace sd {
namespace ops {
namespace helpers {


void bgemm(NDArray *a,  NDArray *b,  NDArray *c,   NDArray *alphas,   NDArray *betas,
           int transA, int transB, int M, int N, int K,  int lda,  int ldb,  int ldc,
           NDArray *all) {
  NDArray *allIndex = nullptr;
  if(all != nullptr)
    allIndex = all;
  else {
    NDArray *allLocal = NDIndexUtils::createAll();
    allIndex = allLocal;
  }

  int batchSize = a->sizeAt(0);
  std::vector<NDArray *>inputs;
  std::vector<NDArray *> bInputs;
  std::vector<NDArray *> outputs;

  ops::create_view createView;

  //divide by 2: queries and keys
  for(int i = 0; i < batchSize; i++) {
    auto point = NDIndexUtils::createPoint(i);
    auto aSlice = createView.evaluate({a,point,allIndex,allIndex},{},{});
    auto bSlice = createView.evaluate({b,point,allIndex,allIndex},{},{});
    auto outSlice = createView.evaluate({c,point,allIndex,allIndex},{},{});
    inputs.push_back(aSlice.at(0));
    bInputs.push_back(bSlice.at(0));
    outputs.push_back(outSlice.at(0));
    delete point;
  }

  delete allIndex;



  bgemm(inputs, bInputs,outputs,alphas,betas,transA,transB,M,N,K,lda,ldb,ldc);

}

// Whether m lies column-major and dense, as BLAS reads it with leading dimension m's row count.
static bool isColumnMajorDense(NDArray *m) {
  return m->rankOf() == 2 && m->ordering() == 'f' && m->strideAt(0) == 1 && m->strideAt(1) == m->sizeAt(0);
}

template <typename T>
static void bgemm_( std::vector<NDArray *> &vA,  std::vector<NDArray *> &vB, std::vector<NDArray *> &vC,
                    NDArray *alphas,  NDArray *betas, int transA, int transB, int M, int N, int K,
                    int lda,  int ldb,  int ldc) {
  int batchSize = vA.size();

  // The leading dimensions are BLAS's column-major ones (lda is A's row count), so every matrix is read in
  // 'f' order, as the CUDA helper reads it: an operand laid out otherwise is copied to 'f' first, and an output
  // laid out otherwise receives its 'f' result afterwards. 'c' matrices, SameDiff's default, were read as
  // their transposes.
  std::vector<NDArray *> pA(batchSize), pB(batchSize), pC(batchSize);
  for (int e = 0; e < batchSize; e++) {
    pA[e] = isColumnMajorDense(vA[e]) ? vA[e] : vA[e]->dup('f');
    pB[e] = isColumnMajorDense(vB[e]) ? vB[e] : vB[e]->dup('f');
    pC[e] = isColumnMajorDense(vC[e]) ? vC[e] : vC[e]->dup('f');
  }

  // Use batched BLAS only when: 1) batched GEMM is available AND 2) BLAS is enabled
  // Previously used || which incorrectly entered BLAS path when BLAS was disabled
  if (BlasHelper::getInstance().hasBatchedGEMM<T>() && sd::env_isEnableBlas()) {
    auto arr = vA.at(0);
    CBLAS_TRANSPOSE *tA, *tB;
    int *tM, *tN, *tK, *tldA, *tldB, *tldC, *tsize;
    // mkl requires mnk etc as arrays, cuda doesn't
    ALLOCATE(tA, arr->getContext()->getWorkspace(), batchSize, CBLAS_TRANSPOSE);
    ALLOCATE(tB, arr->getContext()->getWorkspace(), batchSize, CBLAS_TRANSPOSE);
    ALLOCATE(tM, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tN, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tK, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tldA, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tldB, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tldC, arr->getContext()->getWorkspace(), batchSize, int);
    ALLOCATE(tsize, arr->getContext()->getWorkspace(), batchSize, int);

    shape::fill(tA, (CBLAS_TRANSPOSE)transA, batchSize);
    shape::fill(tB, (CBLAS_TRANSPOSE)transB, batchSize);

    shape::fill(tM, M, batchSize);
    shape::fill(tN, N, batchSize);
    shape::fill(tK, K, batchSize);
    shape::fill(tldA, lda, batchSize);
    shape::fill(tldB, ldb, batchSize);
    shape::fill(tldC, ldc, batchSize);
    shape::fill(tsize, 1, batchSize);

    std::vector<T *> buffersA;
    std::vector<T *> buffersB;
    std::vector<T *> buffersC;



    for (int e = 0; e < batchSize; e++) {
      buffersA.push_back(reinterpret_cast<T *>(pA[e]->buffer()));
      buffersB.push_back(reinterpret_cast<T *>(pB[e]->buffer()));
      buffersC.push_back(reinterpret_cast<T *>(pC[e]->buffer()));
    }

    // Acquire BLAS lock to prevent OpenBLAS TLS corruption and race conditions
    auto blasLock = BlasHelper::getInstance().lockBlas();

    // Inside BLAS block, only check type - BLAS enablement was already verified in outer condition
    if (std::is_same<T, double>::value) {
      BlasHelper::getInstance().dgemmBatched()(CblasColMajor, tA, tB, tM, tN, tK, (double *)alphas->buffer(),
                                               (double **)buffersA.data(), tldA, (double **)buffersB.data(), tldB,
                                               (double *)betas->buffer(), (double **)buffersC.data(), tldC, vA.size(),
                                               tsize);
    } else if (std::is_same<T, float>::value) {
      BlasHelper::getInstance().sgemmBatched()(
          CblasColMajor, tA, tB, tM, tN, tK, (float *)alphas->buffer(), (float **)buffersA.data(), tldA,
          (float **)buffersB.data(), tldB, (float *)betas->buffer(), (float **)buffersC.data(), tldC, vA.size(), tsize);
    }



    // release temporary arrays
    RELEASE(tA, arr->getContext()->getWorkspace());
    RELEASE(tB, arr->getContext()->getWorkspace());
    RELEASE(tM, arr->getContext()->getWorkspace());
    RELEASE(tN, arr->getContext()->getWorkspace());
    RELEASE(tK, arr->getContext()->getWorkspace());
    RELEASE(tldA, arr->getContext()->getWorkspace());
    RELEASE(tldB, arr->getContext()->getWorkspace());
    RELEASE(tldC, arr->getContext()->getWorkspace());
    RELEASE(tsize, arr->getContext()->getWorkspace());
  } else {

    // Products accumulate in AccT: float for the 16-bit types, whose own sums lose low bits at every term.
    using AccT = typename simdOps::AggregateType<T>::type;
    CBLAS_TRANSPOSE tA = (CBLAS_TRANSPOSE)transA;
    CBLAS_TRANSPOSE tB = (CBLAS_TRANSPOSE)transB;
    int vaSize = vA.size();
    auto func = PRAGMA_THREADS_FOR {
      for (auto p = start; p < stop; p++) {
        auto A = reinterpret_cast<T *>(pA.at(p)->buffer());
        auto B = reinterpret_cast<T *>(pB.at(p)->buffer());
        auto C = reinterpret_cast<T *>(pC.at(p)->buffer());
        // Handle scalar, single-element, or empty arrays (use defaults for empty)
        auto alpha = (alphas->isScalar() || alphas->lengthOf() <= 1)
                     ? (alphas->lengthOf() > 0 ? alphas->e<AccT>(0) : static_cast<AccT>(1))
                     : alphas->e<AccT>(p);
        auto beta = (betas->isScalar() || betas->lengthOf() <= 1)
                    ? (betas->lengthOf() > 0 ? betas->e<AccT>(0) : static_cast<AccT>(0))
                    : betas->e<AccT>(p);
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            AccT c_mnp = static_cast<AccT>(0);
            PRAGMA_OMP_SIMD
            for (int k = 0; k < K; k++) {
              c_mnp += static_cast<AccT>(A[tA == CblasNoTrans ? (m + k * lda) : (m * lda + k)]) *
                       static_cast<AccT>(B[tB == CblasNoTrans ? (k + n * ldb) : (k * ldb + n)]);
            }
            C[m + n * ldc] = static_cast<T>(alpha * c_mnp + beta * static_cast<AccT>(C[m + n * ldc]));
          }
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, vaSize);

  }

  for (int e = 0; e < batchSize; e++) {
    if (pC[e] != vC[e]) {
      vC[e]->assign(pC[e]);
      delete pC[e];
    }
    if (pA[e] != vA[e]) delete pA[e];
    if (pB[e] != vB[e]) delete pB[e];
  }
}

void bgemm( std::vector<NDArray *> &vA,  std::vector<NDArray *> &vB, std::vector<NDArray *> &vC,
            NDArray *alphas,  NDArray *betas, int transA, int transB, int M, int N, int K,  int lda,
            int ldb,  int ldc) {
  auto xType = vA.at(0)->dataType();
  BUILD_SINGLE_SELECTOR(xType, bgemm_, (vA, vB, vC, alphas, betas, transA, transB, M, N, K, lda, ldb, ldc),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void bgemm_,
                      ( std::vector<NDArray *> &vA,  std::vector<NDArray *> &vB, std::vector<NDArray *> &vC,
                          NDArray *alphas,  NDArray *betas, int transA, int transB, int M, int N, int K,
                          int lda,  int ldb,  int ldc),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif