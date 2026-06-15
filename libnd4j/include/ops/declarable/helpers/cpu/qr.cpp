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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/qr.h>
#if NOT_EXCLUDED(OP_qr)
namespace sd {
namespace ops {
namespace helpers {

// Modified Gram-Schmidt QR decomposition
template <typename T>
void qrSingle(NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatricies) {
  sd::LongType M = matrix->sizeAt(-2);
  sd::LongType N = matrix->sizeAt(-1);
  sd::LongType K = std::min(M, N);
  
  // Initialize Q and R
  sd::LongType Qcols = fullMatricies ? M : K;
  std::vector<sd::LongType> qShape = {M, Qcols};
  std::vector<sd::LongType> rShape = {K, N};
  
  // Copy input to work
  NDArray* work = matrix->dup();
  
  // Modified Gram-Schmidt
  for (sd::LongType j = 0; j < K; j++) {
    // Compute norm of column j
    T norm = T(0);
    for (sd::LongType i = 0; i < M; i++) {
      norm += work->t<T>(i, j) * work->t<T>(i, j);
    }
    norm = math::sd_sqrt<T, T>(norm);
    
    // Handle near-zero norm
    if (norm < T(1e-10)) {
      // Column is linearly dependent, set Q[:,j] = 0 and R[:,j] = 0
      for (sd::LongType i = 0; i < M; i++) {
        Q->r<T>(i, j) = T(0);
      }
      for (sd::LongType k = j; k < N; k++) {
        R->r<T>(j, k) = T(0);
      }
      continue;
    }
    
    // Q[:,j] = work[:,j] / norm
    for (sd::LongType i = 0; i < M; i++) {
      Q->r<T>(i, j) = work->t<T>(i, j) / norm;
    }
    
    // R[j,j] = norm
    R->r<T>(j, j) = norm;
    
    // Orthogonalize remaining columns
    for (sd::LongType k = j + 1; k < N; k++) {
      // R[j,k] = Q[:,j]^T * work[:,k]
      T dot = T(0);
      for (sd::LongType i = 0; i < M; i++) {
        dot += Q->t<T>(i, j) * work->t<T>(i, k);
      }
      R->r<T>(j, k) = dot;
      
      // work[:,k] -= R[j,k] * Q[:,j]
      for (sd::LongType i = 0; i < M; i++) {
        work->r<T>(i, k) -= dot * Q->t<T>(i, j);
      }
    }
  }
  
  // If full matrices and M > N, complete Q with additional orthogonal vectors
  if (fullMatricies && M > N) {
    // For now, just set remaining columns to zero
    // A proper implementation would use Gram-Schmidt on random vectors
    for (sd::LongType j = K; j < M; j++) {
      for (sd::LongType i = 0; i < M; i++) {
        Q->r<T>(i, j) = (i == j) ? T(1) : T(0);
      }
    }
  }
  
  delete work;
}

template <typename T>
void qr_(NDArray * input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
  if (input->rankOf() == 2) {
    qrSingle<T>(input, outputQ, outputR, fullMatricies);
    return;
  }

  sd::LongType lastDim = input->rankOf() - 1;
  sd::LongType preLastDim = input->rankOf() - 2;
  ResultSet listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  auto batching = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; batch++) {
      qrSingle<T>(listInput.at(batch), listOutQ.at(batch), listOutR.at(batch), fullMatricies);
    }
  };
  samediff::Threads::parallel_tad(batching, 0, listOutQ.size(), 1);
}

void qr(sd::LaunchContext* context, NDArray * input, NDArray* outputQ, NDArray* outputR,
        bool const fullMatricies) {
  BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (input, outputQ, outputR, fullMatricies), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
