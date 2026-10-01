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
// CPU implementation of the fused element-wise chain kernel.
//

#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <ops/declarable/helpers/fusedElementwiseChainMath.h>

#if NOT_EXCLUDED(OP_fused_elementwise_chain)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void fusedElementwiseChain_(NDArray* input, NDArray* output, const FusedElemOp* ops, int numOps,
                                   NDArray** secondaryInputs, const double* clipMin, const double* clipMax) {
  const LongType length = output->lengthOf();
  const int rank = output->rankOf();
  const LongType* zShape = output->shapeOf();
  const LongType* xStrides = input->stridesOf();
  const LongType* zStrides = output->stridesOf();
  const T* x = input->bufferAsT<T>();
  T* z = output->bufferAsT<T>();

  // Same conversion eager clipbyvalue applies to its double bounds.
  const T clipLow = clipMin != nullptr ? static_cast<T>(*clipMin) : static_cast<T>(0);
  const T clipHigh = clipMax != nullptr ? static_cast<T>(*clipMax) : static_cast<T>(0);

  int codes[FUSED_CHAIN_MAX_OPS] = {};
  const T* secondary[FUSED_CHAIN_MAX_OPS] = {};
  bool secondaryScalar[FUSED_CHAIN_MAX_OPS] = {};
  const LongType* secondaryShape[FUSED_CHAIN_MAX_OPS] = {};
  const LongType* secondaryStrides[FUSED_CHAIN_MAX_OPS] = {};
  int secondaryRank[FUSED_CHAIN_MAX_OPS] = {};

  // Linear mode: every operand holds the element of output index i at offset i (or is a scalar).
  bool linear = isFusedChainDenseC(input) && isFusedChainDenseC(output);
  for (int m = 0; m < numOps; m++) {
    codes[m] = static_cast<int>(ops[m]);
    if (!isBinaryFusedOp(ops[m])) continue;
    NDArray* operand = secondaryInputs[m];
    secondary[m] = operand->bufferAsT<T>();
    secondaryScalar[m] = operand->lengthOf() == 1;
    secondaryShape[m] = operand->shapeOf();
    secondaryStrides[m] = operand->stridesOf();
    secondaryRank[m] = operand->rankOf();
    if (!secondaryScalar[m] && !(operand->isSameShape(output) && isFusedChainDenseC(operand))) linear = false;
  }

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK] = {};
    for (auto i = start; i < stop; i++) {
      LongType xOffset = i;
      LongType zOffset = i;
      if (!linear) {
        INDEX2COORDS(i, rank, zShape, coords);
        COORDS2INDEX(rank, xStrides, coords, xOffset);
        COORDS2INDEX(rank, zStrides, coords, zOffset);
      }

      T value = x[xOffset];
      for (int m = 0; m < numOps; m++) {
        T operand = static_cast<T>(0);
        if (secondary[m] != nullptr) {
          LongType sOffset = 0;
          if (!secondaryScalar[m]) {
            sOffset = linear ? static_cast<LongType>(i)
                             : fusedChainBroadcastOffset(coords, rank, secondaryShape[m], secondaryStrides[m],
                                                         secondaryRank[m]);
          }
          operand = secondary[m][sOffset];
        }
        value = fusedChainStep<T>(codes[m], value, operand, clipLow, clipHigh);
      }
      z[zOffset] = value;
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
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

  NDArray::preparePrimaryUse({output}, reads);
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedElementwiseChain_,
                        (input, output, ops, numOps, secondaryInputs, clipMin, clipMax), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, reads);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
