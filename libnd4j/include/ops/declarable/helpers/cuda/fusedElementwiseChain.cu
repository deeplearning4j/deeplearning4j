/* ******************************************************************************
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
// @author Adam Gibson
//
// CUDA implementation of the fused element-wise chain kernel.
// Processes the entire chain of operations per-element in a single kernel,
// keeping intermediate values in registers instead of global memory. Each
// member still rounds to the storage type, as its materialized output would.
//
// FLOAT32/DOUBLE: nvcc may contract a multiply and a following add from
// adjacent members into one FMA, which a materialized intermediate cannot;
// results can then differ from eager execution in the last bit. HALF and
// BFLOAT16 members round explicitly, so they match eager exactly.
//

#include <cuda_runtime.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <ops/declarable/helpers/fusedElementwiseChainMath.h>

#include <type_traits>

#if NOT_EXCLUDED(OP_fused_elementwise_chain)
namespace sd {
namespace ops {
namespace helpers {

// Codes and secondary operands of one chain, passed to the kernel by value. Shape and stride
// pointers address device shape info.
struct FusedChainOperands {
  int numOps;
  int codes[FUSED_CHAIN_MAX_OPS];
  const void* secondary[FUSED_CHAIN_MAX_OPS];  // nullptr for unary members
  const LongType* secondaryShape[FUSED_CHAIN_MAX_OPS];
  const LongType* secondaryStrides[FUSED_CHAIN_MAX_OPS];
  int secondaryRank[FUSED_CHAIN_MAX_OPS];
  bool secondaryScalar[FUSED_CHAIN_MAX_OPS];
};

static_assert(std::is_trivially_copyable<FusedChainOperands>::value,
              "FusedChainOperands is passed as a CUDA kernel argument");

template <typename T>
static SD_KERNEL void fusedElementwiseChainCuda(const void* vx, const LongType* xStrides, void* vz,
                                                const LongType* zShape, const LongType* zStrides, const int rank,
                                                const LongType length, const bool linear,
                                                const FusedChainOperands operands, const T clipLow,
                                                const T clipHigh) {
  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  LongType coords[SD_MAX_RANK];
  for (LongType linearIndex = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; linearIndex < length;
       linearIndex += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType xOffset = linearIndex;
    LongType zOffset = linearIndex;
    if (!linear) {
      INDEX2COORDS(linearIndex, rank, zShape, coords);
      COORDS2INDEX(rank, xStrides, coords, xOffset);
      COORDS2INDEX(rank, zStrides, coords, zOffset);
    }

    T value = x[xOffset];
    for (int m = 0; m < operands.numOps; m++) {
      T operand = static_cast<T>(0);
      const T* secondary = reinterpret_cast<const T*>(operands.secondary[m]);
      if (secondary != nullptr) {
        LongType sOffset = 0;
        if (!operands.secondaryScalar[m]) {
          sOffset = linear ? linearIndex
                           : fusedChainBroadcastOffset(coords, rank, operands.secondaryShape[m],
                                                       operands.secondaryStrides[m], operands.secondaryRank[m]);
        }
        operand = secondary[sOffset];
      }
      value = fusedChainStep<T>(operands.codes[m], value, operand, clipLow, clipHigh);
    }
    z[zOffset] = value;
  }
}

template <typename T>
static void fusedElementwiseChain_(LaunchContext* context, NDArray* input, NDArray* output, const FusedElemOp* ops,
                                   int numOps, NDArray** secondaryInputs, const double* clipMin,
                                   const double* clipMax) {
  const LongType length = output->lengthOf();
  const int rank = output->rankOf();

  FusedChainOperands operands{};
  operands.numOps = numOps;
  // Linear mode: every operand holds the element of output index i at offset i (or is a scalar).
  bool linear = isFusedChainDenseC(input) && isFusedChainDenseC(output);
  for (int m = 0; m < numOps; m++) {
    operands.codes[m] = static_cast<int>(ops[m]);
    if (!isBinaryFusedOp(ops[m])) continue;
    NDArray* operand = secondaryInputs[m];
    const int operandRank = operand->rankOf();
    operands.secondary[m] = operand->specialBuffer();
    operands.secondaryShape[m] = operand->specialShapeInfo() + 1;
    operands.secondaryStrides[m] = operand->specialShapeInfo() + 1 + operandRank;
    operands.secondaryRank[m] = operandRank;
    operands.secondaryScalar[m] = operand->lengthOf() == 1;
    if (!operands.secondaryScalar[m] && !(operand->isSameShape(output) && isFusedChainDenseC(operand))) linear = false;
  }

  // Same conversion eager clipbyvalue applies to its double bounds.
  const T clipLow = clipMin != nullptr ? static_cast<T>(*clipMin) : static_cast<T>(0);
  const T clipHigh = clipMax != nullptr ? static_cast<T>(*clipMax) : static_cast<T>(0);

  dim3 launchDims = getLaunchDims("fused_elementwise_chain");
  if (launchDims.x == 0 || launchDims.y == 0 || launchDims.y > 1024) {
    THROW_EXCEPTION("fused_elementwise_chain: GRID_SIZE_FUSED_ELEMENTWISE_CHAIN must be positive and "
                    "BLOCK_SIZE_FUSED_ELEMENTWISE_CHAIN within 1..1024");
  }
  const LongType threads = launchDims.y;
  LongType blocks = (length + threads - 1) / threads;
  if (blocks > static_cast<LongType>(launchDims.x)) blocks = launchDims.x;

  auto stream = context->getCudaStream();
  fusedElementwiseChainCuda<T><<<static_cast<unsigned int>(blocks), launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo() + 1 + rank, output->specialBuffer(),
      output->specialShapeInfo() + 1, output->specialShapeInfo() + 1 + rank, rank, length, linear, operands,
      clipLow, clipHigh);
  DebugHelper::checkGlobalErrorCode("fusedElementwiseChainCuda failed");
}

void fusedElementwiseChain(NDArray* input, NDArray* output, const FusedElemOp* ops, int numOps,
                           NDArray** secondaryInputs, const double* clipMin, const double* clipMax,
                           LaunchContext* context) {
  const std::string reason =
      fusedChainUnsupportedReason(input, output, ops, numOps, secondaryInputs, clipMin, clipMax);
  if (!reason.empty()) {
    THROW_EXCEPTION(("fused_elementwise_chain: " + reason).c_str());
  }
  if (output->isEmpty()) return;

  std::vector<NDArray*> reads = {input};
  for (int m = 0; m < numOps; m++) {
    if (isBinaryFusedOp(ops[m])) reads.push_back(secondaryInputs[m]);
  }

  NDArray::prepareSpecialUse({output}, reads);
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedElementwiseChain_,
                        (context, input, output, ops, numOps, secondaryInputs, clipMin, clipMax), SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, reads);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
