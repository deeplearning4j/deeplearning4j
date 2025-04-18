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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/qr.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void matrixMinorKernel(T* outBuffer, LongType* outShape, T* inBuffer, LongType* inShape,
                                        LongType column, LongType rows, LongType columns) {

  for (auto i = blockIdx.x; i < rows; i += gridDim.x)
    for (auto j = threadIdx.x; j < columns; j += blockDim.x) {
      LongType pos[] = {i, j};

      LongType zIndex;
      COORDS2INDEX(shape::rank(outShape), shape::stride(outShape), pos, zIndex);

      LongType xIndex;
      COORDS2INDEX(shape::rank(inShape), shape::stride(inShape), pos, xIndex);

      if (i < column || j < column) {
        outBuffer[zIndex] = i != j ? T(0.f) : T(1.f);
      } else {
        outBuffer[zIndex] = inBuffer[xIndex];  // m.t<T>(i,j) = in.t<T>(i,j);
      }
    }
}

template <typename T>
NDArray matrixMinor(LaunchContext* context, NDArray& in, LongType col) {
  NDArray *m = in.ulike();
  m->setIdentity();
  NDArray view = *m;
  NDArray assign = in({col, m->rows(), col, m->columns()});
  view({col, m->rows(), col, m->columns()}).assign(&assign);

  m->tickWriteDevice();
  return *m;
}

/* m = I - v v^T */
template <typename T>
static SD_KERNEL void vmulKernel(T* resBuf, const LongType* resShape, T const* vBuff, LongType const* vShape,
                                 LongType n) {
  for (auto i = blockIdx.x; i < n; i += gridDim.x)
    for (auto j = threadIdx.x; j < n; j += blockDim.x) {
      LongType posR[] = {i, j};
      LongType indexR, indexX, indexY;
      COORDS2INDEX(shape::rank(resShape), shape::stride(resShape), posR, indexR);
      COORDS2INDEX(1, shape::stride(vShape), &i, indexX);
      COORDS2INDEX(1, shape::stride(vShape), &j, indexY);

      resBuf[indexR] = T(-2.f) * vBuff[indexX] * vBuff[indexY] + (i != j ? T(0.f) : T(1.f));
    }
}

template <typename T>
NDArray vmul(LaunchContext* context, NDArray& v, int n) {
  std::vector<LongType> shape = {n, n};
  NDArray res('c', shape, v.dataType(), context);  // x = matrix_new(n, n);

  auto stream = context->getCudaStream();
  dim3 launchDims = getLaunchDims("qr");
  vmulKernel<T><<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(res.dataBuffer()->specialAsT<T>(), res.specialShapeInfo(),
                                            reinterpret_cast<T const*>(v.specialBuffer()), v.specialShapeInfo(), n);
  sd::DebugHelper::checkErrorCode(stream, "vmulKernel failed");

  return res;
}

template <typename T>
static bool diagonalIsPositive(NDArray* matrix, LongType k) {
  T hVal;
  LongType pos[] = {k, k};
  LongType shift;
  COORDS2INDEX(shape::rank(matrix->shapeInfo()), shape::stride(matrix->shapeInfo()), pos, shift);
  cudaMemcpy(&hVal, matrix->specialBuffer(), sizeof(T), cudaMemcpyDeviceToHost);
  return hVal > T(0.f);
}

template <typename T>
void qrSingle(LaunchContext* context, NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatrices) {
  LongType M = matrix->sizeAt(0);
  LongType N = matrix->sizeAt(1);
  auto resQ = fullMatrices ? *Q->ulike() : NDArrayFactory::create<T>(matrix->ordering(), {M, M}, Q->getContext());
  auto resR = fullMatrices ? R->ulike() : matrix->ulike();
  std::vector<NDArray> q(M);
  NDArray z = *matrix;
  std::vector<LongType> shape = {M};

  NDArray e('c', shape, DataTypeUtils::fromT<T>(), context);  // two internal buffers and scalar for squared norm
  for (auto k = 0; k < N && k < M - 1; k++) {               // loop for columns, but not further then row number
    e.nullify();
    z = matrixMinor<T>(context, z,
                       k);  // minor computing for current column with given matrix z (initally is a input matrix)

    auto currentColumn = z({0, 0, k, k + 1});  // retrieve k column from z to x buffer
    std::vector<LongType> zero = {0};
    auto norm = currentColumn.reduceAlongDimension(reduce::Norm2, &zero);
    if (diagonalIsPositive<T>(matrix, k))  // matrix->t<T>(k,k) > T(0.f)) // negate on positive matrix diagonal element
      norm.applyTransform(transform::Neg, &norm);  // *= -1.f;//-norm.t<T>(0);

    e.p(k, &norm);        // e - is filled by 0 vector except diagonal element (filled by 1)
    e += currentColumn;  // e[i] = x[i] + a * e[i] for each i from 0 to n - 1
    auto normE = e.reduceAlongDimension(reduce::Norm2, &zero);
    e /= normE;
    q[k] = vmul<T>(context, e, M);
    auto qQ = z.ulike();
    MmulHelper::matmul(&q[k], &z, qQ, false, false,1.0,0.0,qQ);
    z = std::move(*qQ);
  }
  resQ.assign(&q[0]);

  for (int i = 1; i < N && i < M - 1; i++) {
    auto tempResQ = resQ;
    MmulHelper::matmul(&q[i],&resQ, &tempResQ, false, false,1.0,0.0,&tempResQ);
    resQ = std::move(tempResQ);
  }
  MmulHelper::matmul(&resQ, matrix, resR, false, false,1.0,0.0,resR);
  // resR *= -1.f;
  resQ.transposei();

  if (fullMatrices) {
    Q->assign(&resQ);
    R->assign(resR);
  } else {
    NDArray resRRef = *resR;
    NDArray qAssign = resQ({0, 0, 0, N});
    Q->assign(&qAssign);
    NDArray rAssign = resRRef({0, N, 0, 0});
    R->assign(&rAssign);
  }
}

template <typename T>
void qr_(LaunchContext* context, NDArray * input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
  LongType lastDim = input->rankOf() - 1;
  LongType preLastDim = input->rankOf() - 2;

  NDArray::prepareSpecialUse({outputQ, outputR}, {input});
  ResultSet listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  ResultSet listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
  auto start = 0;
  auto stop = listInput.size();
  auto increment = 1;

  for (auto batch = start; batch < stop; batch += increment) {
    // qr here
    qrSingle<T>(context, listInput.at(batch), listOutQ.at(batch), listOutR.at(batch), fullMatricies);
  }
  NDArray::registerSpecialUse({outputQ, outputR}, {input});
}

void qr(LaunchContext* context, NDArray * input, NDArray* outputQ, NDArray* outputR,
        bool const fullMatricies) {
  BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (context, input, outputQ, outputR, fullMatricies), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
