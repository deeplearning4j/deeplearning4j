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

template <typename T>
NDArray matrixMinor(NDArray& in, sd::LongType col) {
  NDArray *m = in.ulike();
  m->setIdentity();
  auto mRef = *m;
  auto view =  mRef({col, m->rows(), col, m->columns()});
  view.assign(in({col, m->rows(), col, m->columns()}));

  return mRef;
}

/* m = I - v v^T */
template <typename T>
NDArray vmul(NDArray& v, int n) {
  std::vector<sd::LongType> nShape = {n,n};
  NDArray res('c', nShape, v.dataType(), v.getContext());  // x = matrix_new(n, n);
  T const* vBuf = v.getDataBuffer()->primaryAsT<T>();
  T* resBuf = res.dataBuffer()->primaryAsT<T>();
  auto interloop = PRAGMA_THREADS_FOR_2D {
    for (auto i = start_x; i < n; i += inc_x)
      for (auto j = start_y; j < n; j += inc_y) resBuf[i * n + j] = -2 * vBuf[i] * vBuf[j] + (i == j ? T(1) : T(0));
  };

  samediff::Threads::parallel_for(interloop, 0, n, 1, 0, n, 1);
  return res;
}

template <typename T>
void qrSingle(NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatricies) {
  sd::LongType M = matrix->sizeAt(-2);
  sd::LongType N = matrix->sizeAt(-1);
  auto resQ = fullMatricies ? Q->ulike() : new NDArray(NDArrayFactory::create<T>(matrix->ordering(), {M, M}, Q->getContext()));
  auto resR = fullMatricies ? R->ulike() : matrix->ulike();
  std::vector<NDArray> q(M);

  std::vector<sd::LongType> mShape = {M};
  NDArray z = *matrix;
  NDArray e('c', mShape, DataTypeUtils::fromT<T>(), Q->getContext());  // two internal buffers and scalar for squared norm

  for (sd::LongType k = 0; k < N && k < M - 1; k++) {  // loop for columns, but not further then row number
    e.nullify();
    z = matrixMinor<T>(z, k);  // minor computing for current column with given matrix z (initally is a input matrix)

    std::vector<sd::LongType> zeroVec = {0};
    auto currentColumn = z({0, 0, k, k + 1});  // retrieve k column from z to x buffer
    auto norm = currentColumn.reduceAlongDimension(reduce::Norm2,&zeroVec);
    if (matrix->t<T>(k, k) > T(0.f))  // negate on positive matrix diagonal element
      norm *= T(-1.f);


    e.p(k, &norm);
    e += currentColumn;  //  e += tE; // e[i] = x[i] + a * e[i] for each i from 0 to n - 1
    auto normE = e.reduceAlongDimension(reduce::Norm2, &zeroVec);
    e /= normE;
    q[k] = vmul<T>(e, M);
    auto qQ = z.ulike();
    MmulHelper::matmul(&q[k], &z, qQ, false, false, 0, 0, qQ);
    z = std::move(*qQ);
  }
  resQ->assign(q[0]);  //

  for (sd::LongType i = 1; i < N && i < M - 1; i++) {
    auto tempResQ = resQ;
    MmulHelper::matmul(&q[i], resQ, tempResQ, false, false, 0, 0, tempResQ);  // use mmulMxM?
    resQ = std::move(tempResQ);
  }
  MmulHelper::matmul(resQ, matrix, resR, false, false, 0, 0, resR);
  // resR *= -1.f;
  resQ->transposei();
  if (fullMatricies) {
    Q->assign(resQ);
    R->assign(resR);
  } else {
    auto resQRef = *resQ;
    auto resRRef = *resR;
    auto resQView = resQRef({0, 0, 0, N});
    Q->assign(&resQRef({0, 0, 0, N}));
    R->assign(&resRRef({0, N, 0, 0}));
  }

  delete resQ;
  delete resR;
}

template <typename T>
void qr_(NDArray * input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
  sd::LongType lastDim = input->rankOf() - 1;
  sd::LongType preLastDim = input->rankOf() - 2;
  ResultSet listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  auto batching = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; batch++) {
      // qr here
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