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
// Mamba-2 SSD recurrence — CPU implementation
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/mamba2_ssm.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void mamba2Ssm_(LaunchContext* context,
                       NDArray* x, NDArray* A, NDArray* B, NDArray* C,
                       NDArray* dt, NDArray* D, NDArray* stateIn,
                       NDArray* y, NDArray* stateOut,
                       int H, int P, int N) {
  const LongType Batch = x->sizeAt(0);
  const LongType L     = x->sizeAt(1);

  const T* xBuf  = x->bufferAsT<T>();
  const T* aBuf  = A->bufferAsT<T>();
  const T* bBuf  = B->bufferAsT<T>();
  const T* cBuf  = C->bufferAsT<T>();
  const T* dtBuf = dt->bufferAsT<T>();
  T* yBuf        = y->bufferAsT<T>();
  T* soBuf       = stateOut->bufferAsT<T>();

  // x strides: [B, L, D]
  const auto xS0 = x->strideAt(0), xS1 = x->strideAt(1), xS2 = x->strideAt(2);
  // A strides: [H]
  const auto aS0 = A->strideAt(0);
  // B strides: [B, L, N]
  const auto bS0 = B->strideAt(0), bS1 = B->strideAt(1), bS2 = B->strideAt(2);
  // C strides: [B, L, N]
  const auto cS0 = C->strideAt(0), cS1 = C->strideAt(1), cS2 = C->strideAt(2);
  // dt strides: [B, L, H]
  const auto dtS0 = dt->strideAt(0), dtS1 = dt->strideAt(1), dtS2 = dt->strideAt(2);
  // y strides: [B, L, D]
  const auto yS0 = y->strideAt(0), yS1 = y->strideAt(1), yS2 = y->strideAt(2);
  // stateOut strides: [B, H, N, P]
  const auto soS0 = stateOut->strideAt(0), soS1 = stateOut->strideAt(1);
  const auto soS2 = stateOut->strideAt(2), soS3 = stateOut->strideAt(3);

  // D (skip connection) — optional
  const T* dBuf = nullptr;
  LongType dS0 = 0;
  if (D != nullptr) {
    dBuf = D->bufferAsT<T>();
    dS0 = D->strideAt(0);
  }

  // Working state buffer: contiguous [B, H, N, P]
  const LongType stateElems = Batch * H * N * P;
  std::vector<T> state(stateElems, static_cast<T>(0));

  // Copy stateIn if provided
  if (stateIn != nullptr) {
    const T* siBuf = stateIn->bufferAsT<T>();
    const auto siS0 = stateIn->strideAt(0), siS1 = stateIn->strideAt(1);
    const auto siS2 = stateIn->strideAt(2), siS3 = stateIn->strideAt(3);
    for (LongType b = 0; b < Batch; ++b)
      for (LongType h = 0; h < H; ++h)
        for (LongType n = 0; n < N; ++n)
          for (LongType p = 0; p < P; ++p)
            state[((b * H + h) * N + n) * P + p] =
                siBuf[b * siS0 + h * siS1 + n * siS2 + p * siS3];
  }

  // Sequential over timesteps (recurrence), parallel over batch * heads
  for (LongType t = 0; t < L; ++t) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto bh = start; bh < stop; ++bh) {
        const LongType b = bh / H;
        const LongType h = bh % H;

        // A_bar = exp(A[h] * dt[b,t,h])
        const T logA = aBuf[h * aS0];
        const T dtVal = dtBuf[b * dtS0 + t * dtS1 + h * dtS2];
        const T aBar = sd::math::sd_exp<T, T>(logA * dtVal);

        T* sPtr = state.data() + ((b * H + h) * N) * P;

        // State update: for each (n, p)
        //   state[n,p] = aBar * state[n,p] + B[b,t,n] * dt * x[b,t,h*P+p]
        for (LongType n = 0; n < N; ++n) {
          const T bVal = bBuf[b * bS0 + t * bS1 + n * bS2];
          const T bDt = bVal * dtVal;
          for (LongType p = 0; p < P; ++p) {
            const T xVal = xBuf[b * xS0 + t * xS1 + (h * P + p) * xS2];
            sPtr[n * P + p] = aBar * sPtr[n * P + p] + bDt * xVal;
          }
        }

        // Output: y[b,t,h*P+p] = sum_n(C[b,t,n] * state[n,p])
        for (LongType p = 0; p < P; ++p) {
          T yVal = static_cast<T>(0);
          for (LongType n = 0; n < N; ++n) {
            const T cVal = cBuf[b * cS0 + t * cS1 + n * cS2];
            yVal += cVal * sPtr[n * P + p];
          }

          // Skip connection: y += D[h] * x[b,t,h*P+p]
          if (dBuf != nullptr) {
            const T xVal = xBuf[b * xS0 + t * xS1 + (h * P + p) * xS2];
            yVal += dBuf[h * dS0] * xVal;
          }

          yBuf[b * yS0 + t * yS1 + (h * P + p) * yS2] = yVal;
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, Batch * H);
  }

  // Copy final state to stateOut
  for (LongType b = 0; b < Batch; ++b)
    for (LongType h = 0; h < H; ++h)
      for (LongType n = 0; n < N; ++n)
        for (LongType p = 0; p < P; ++p)
          soBuf[b * soS0 + h * soS1 + n * soS2 + p * soS3] =
              state[((b * H + h) * N + n) * P + p];
}

void mamba2Ssm(LaunchContext* context,
               NDArray* x, NDArray* A, NDArray* B, NDArray* C,
               NDArray* dt, NDArray* D, NDArray* stateIn,
               NDArray* y, NDArray* stateOut,
               int numHeads, int headDim, int stateDim) {
  NDArray::preparePrimaryUse({y, stateOut}, {x, A, B, C, dt});
  if (D != nullptr) NDArray::preparePrimaryUse({}, {D});
  if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

  BUILD_SINGLE_SELECTOR(x->dataType(), mamba2Ssm_,
      (context, x, A, B, C, dt, D, stateIn, y, stateOut, numHeads, headDim, stateDim),
      SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({y, stateOut}, {x, A, B, C, dt});
  if (D != nullptr) NDArray::registerPrimaryUse({}, {D});
  if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
