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
// Mamba-2 SSD recurrence — CUDA implementation
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <types/float16.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/mamba2_ssm.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel: one block per (batch, head).
// Threads parallelize over (n, p) state dimensions.
// Time steps are sequential (launched per-timestep from host).
///////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void mamba2SsmKernel(
    const T* __restrict__ x,
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ C,
    const T* __restrict__ dt,
    const T* __restrict__ D,     // may be nullptr
    T* __restrict__ state,       // [B, H, N, P] contiguous working state
    T* __restrict__ y,
    const LongType Batch, const LongType H, const LongType N, const LongType P,
    const LongType t,            // current timestep
    const int hasD,              // 1 if D is non-null
    // x strides [B, L, D]
    const LongType xS0, const LongType xS1, const LongType xS2,
    // A strides [H]
    const LongType aS0,
    // B strides [B, L, N]
    const LongType bS0, const LongType bS1, const LongType bS2,
    // C strides [B, L, N]
    const LongType cS0, const LongType cS1, const LongType cS2,
    // dt strides [B, L, H]
    const LongType dtS0, const LongType dtS1, const LongType dtS2,
    // D strides [H]
    const LongType dS0,
    // y strides [B, L, D]
    const LongType yS0, const LongType yS1, const LongType yS2) {

  const LongType bh = blockIdx.x;
  if (bh >= Batch * H) return;

  const LongType b = bh / H;
  const LongType h = bh % H;

  // A_bar = exp(A[h] * dt[b,t,h])
  const T logA = A[h * aS0];
  const T dtVal = dt[b * dtS0 + t * dtS1 + h * dtS2];
  const T aBar = sd::math::sd_exp<T, T>(logA * dtVal);

  T* sPtr = state + ((b * H + h) * N) * P;
  const LongType NP = N * P;

  // Each thread handles a subset of (n, p) pairs via grid-stride within block
  for (LongType np = threadIdx.x; np < NP; np += blockDim.x) {
    const LongType n = np / P;
    const LongType p = np % P;

    const T bVal = B[b * bS0 + t * bS1 + n * bS2];
    const T xVal = x[b * xS0 + t * xS1 + (h * P + p) * xS2];

    // State update: state[n,p] = aBar * state[n,p] + B[b,t,n] * dt * x[b,t,h*P+p]
    sPtr[np] = aBar * sPtr[np] + bVal * dtVal * xVal;
  }

  __syncthreads();

  // Output computation: y[b,t,h*P+p] = sum_n(C[b,t,n] * state[n,p]) + D[h] * x[b,t,h*P+p]
  // Each thread handles a subset of p values
  for (LongType p = threadIdx.x; p < P; p += blockDim.x) {
    T yVal = static_cast<T>(0);
    for (LongType n = 0; n < N; ++n) {
      const T cVal = C[b * cS0 + t * cS1 + n * cS2];
      yVal += cVal * sPtr[n * P + p];
    }

    // Skip connection
    if (hasD) {
      const T xVal = x[b * xS0 + t * xS1 + (h * P + p) * xS2];
      yVal += D[h * dS0] * xVal;
    }

    y[b * yS0 + t * yS1 + (h * P + p) * yS2] = yVal;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Launcher — sequential over timesteps, parallel within each step
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void launchMamba2Ssm(
    const T* x, const T* A, const T* B, const T* C, const T* dt, const T* D,
    T* state, T* y,
    LongType Batch, LongType L, LongType H, LongType N, LongType P,
    int hasD,
    LongType xS0, LongType xS1, LongType xS2,
    LongType aS0,
    LongType bS0, LongType bS1, LongType bS2,
    LongType cS0, LongType cS1, LongType cS2,
    LongType dtS0, LongType dtS1, LongType dtS2,
    LongType dS0,
    LongType yS0, LongType yS1, LongType yS2,
    cudaStream_t stream) {

  const int numBlocks = static_cast<int>(Batch * H);
  const LongType NP = N * P;

  // Size threads to cover (N * P) elements, rounded to warp size
  int threadsPerBlock = 256;
  if (NP < threadsPerBlock) {
    threadsPerBlock = static_cast<int>(((NP + 31) / 32) * 32);
    if (threadsPerBlock < 32) threadsPerBlock = 32;
  }

  for (LongType t = 0; t < L; ++t) {
    mamba2SsmKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        x, A, B, C, dt, D, state, y,
        Batch, H, N, P, t, hasD,
        xS0, xS1, xS2, aS0,
        bS0, bS1, bS2, cS0, cS1, cS2,
        dtS0, dtS1, dtS2, dS0,
        yS0, yS1, yS2);
  }
  DebugHelper::checkGlobalErrorCode("mamba2SsmKernel failed");
}

///////////////////////////////////////////////////////////////////////////////
// Public interface
///////////////////////////////////////////////////////////////////////////////
void mamba2Ssm(LaunchContext* context,
               NDArray* x, NDArray* A, NDArray* B, NDArray* C,
               NDArray* dt, NDArray* D, NDArray* stateIn,
               NDArray* y, NDArray* stateOut,
               int numHeads, int headDim, int stateDim) {
  const LongType Batch = x->sizeAt(0);
  const LongType L     = x->sizeAt(1);
  const LongType H     = static_cast<LongType>(numHeads);
  const LongType P     = static_cast<LongType>(headDim);
  const LongType N     = static_cast<LongType>(stateDim);

  NDArray::prepareSpecialUse({y, stateOut}, {x, A, B, C, dt});
  if (D != nullptr) NDArray::prepareSpecialUse({}, {D});
  if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

  auto stream = context->getCudaStream();

  // Initialize stateOut to zero, then copy stateIn if provided
  stateOut->nullify();
  if (stateIn != nullptr) stateOut->assign(stateIn);

  const int hasD = (D != nullptr) ? 1 : 0;
  const LongType dS0 = (D != nullptr) ? D->strideAt(0) : 0;

  auto dtype = x->dataType();

  #define LAUNCH_MAMBA2(T_TYPE, T_CAST) \
    launchMamba2Ssm<T_TYPE>( \
        reinterpret_cast<const T_TYPE*>(x->specialBuffer()), \
        reinterpret_cast<const T_TYPE*>(A->specialBuffer()), \
        reinterpret_cast<const T_TYPE*>(B->specialBuffer()), \
        reinterpret_cast<const T_TYPE*>(C->specialBuffer()), \
        reinterpret_cast<const T_TYPE*>(dt->specialBuffer()), \
        D != nullptr ? reinterpret_cast<const T_TYPE*>(D->specialBuffer()) : nullptr, \
        reinterpret_cast<T_TYPE*>(stateOut->specialBuffer()), \
        reinterpret_cast<T_TYPE*>(y->specialBuffer()), \
        Batch, L, H, N, P, hasD, \
        x->strideAt(0), x->strideAt(1), x->strideAt(2), \
        A->strideAt(0), \
        B->strideAt(0), B->strideAt(1), B->strideAt(2), \
        C->strideAt(0), C->strideAt(1), C->strideAt(2), \
        dt->strideAt(0), dt->strideAt(1), dt->strideAt(2), \
        dS0, \
        y->strideAt(0), y->strideAt(1), y->strideAt(2), \
        *stream)

  if (dtype == DataType::FLOAT32) {
    LAUNCH_MAMBA2(float, float);
  } else if (dtype == DataType::DOUBLE) {
    LAUNCH_MAMBA2(double, double);
  } else if (dtype == DataType::HALF) {
    LAUNCH_MAMBA2(float16, float16);
  } else {
    THROW_EXCEPTION("mamba2Ssm: Unsupported data type");
  }

  #undef LAUNCH_MAMBA2

  NDArray::registerSpecialUse({y, stateOut}, {x, A, B, C, dt});
  if (D != nullptr) NDArray::registerSpecialUse({}, {D});
  if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
