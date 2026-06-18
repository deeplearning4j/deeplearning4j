/*******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
//
// @author AbdelRauf (original CPU impl)
// @author CUDA port
//
#include <array/DataTypeUtils.h>
#include <array/NDArrayFactory.h>
#include <helpers/PointersManager.h>
#include <math/platformmath.h>
#include <ops/declarable/helpers/ctc.h>
#include <system/op_boilerplate.h>

#include <cmath>
#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {

// ============================================================================
// Device-side helper functions (mirrors of the header-only CPU helpers)
// ============================================================================

template <typename T>
SD_DEVICE SD_INLINE T d_negative_infinity() {
  return -DataTypeUtils::infOrMax<T>();
}

template <typename T>
SD_DEVICE SD_INLINE T d_local_log(T x) {
  if (x > T(0)) {
    return math::p_log<T>(x);
  }
  return d_negative_infinity<T>();
}

template <typename T>
SD_DEVICE SD_INLINE T d_log_sum_exp(T x1, T x2) {
  if (x1 >= x2) {
    return x1 + d_local_log<T>(T(1) + math::p_exp<T>(x2 - x1));
  }
  return x2 + d_local_log<T>(T(1) + math::p_exp<T>(x1 - x2));
}

template <typename T>
SD_DEVICE SD_INLINE T d_log_sum_exp3(T arg1, T arg2, T arg3) {
  T c_max = arg1 > arg2 ? arg1 : arg2;
  c_max = c_max > arg3 ? c_max : arg3;
  if (d_negative_infinity<T>() == c_max) {
    c_max = T(0);
  }
  return math::p_log<T>(math::p_exp<T>(arg1 - c_max) +
                         math::p_exp<T>(arg2 - c_max) +
                         math::p_exp<T>(arg3 - c_max)) + c_max;
}

// ============================================================================
// Device-side CTC forward pass for a single batch element
// Returns -log(P(labels|logits)), i.e. the CTC loss
// ============================================================================

template <typename Type, typename IndexType>
SD_DEVICE Type ctcForwardDevice(
    Type* alphaPtr,           // workspace [lenT * lenSB], pre-filled with negInf
    const Type* logP,         // logits pointer for this batch: &logits[b, 0, 0]
    LongType incP,            // stride along time dim (logits.stridesOf()[1])
    const IndexType* lbl,     // label pointer for this batch: &labels[b, 0]
    LongType elwiseP,         // stride along class dim (logits.stridesOf()[2])
    LongType elwiseS,         // stride along label dim (labels.stridesOf()[1])
    LongType lenSB,           // 2*lenS + 1
    LongType lenT,            // actual input length for this batch
    int blankIndex) {

  Type negInf = d_negative_infinity<Type>();
  LongType incA = lenSB;  // alpha row stride = lenSB (contiguous workspace)

  // Initialize alpha at t=0
  alphaPtr[0] = logP[blankIndex * elwiseP];
  alphaPtr[1] = logP[lbl[0] * elwiseP];
  // rest already negInf

  Type* alphaPrevPtr = alphaPtr;
  alphaPtr += incA;
  logP += incP;

  LongType startX = lenSB - 2 * lenT;

  for (LongType t = 1; t < lenT; t++) {
    LongType s = startX + 2 * t;
    if (s < 0) s = 0;

    for (; s < lenSB; s++) {
      LongType ind = s / 2;
      IndexType currentInd = (s % 2 == 0) ? (IndexType)blankIndex : lbl[ind * elwiseS];

      Type alphaS = alphaPrevPtr[s];
      Type alphaS_1 = s > 0 ? alphaPrevPtr[s - 1] : negInf;
      Type currentProb = logP[currentInd * elwiseP];

      if (s > 1 && currentInd != (IndexType)blankIndex &&
          currentInd != lbl[(ind - 1) * elwiseS]) {
        Type alphaS_2 = alphaPrevPtr[s - 2];
        alphaPtr[s] = d_log_sum_exp3<Type>(alphaS, alphaS_1, alphaS_2) + currentProb;
      } else {
        alphaPtr[s] = d_log_sum_exp<Type>(alphaS, alphaS_1) + currentProb;
      }
    }

    alphaPrevPtr = alphaPtr;
    logP += incP;
    alphaPtr += incA;
  }

  Type logP0 = alphaPrevPtr[lenSB - 1];
  Type logP1 = alphaPrevPtr[lenSB - 2];
  return -d_log_sum_exp<Type>(logP0, logP1);
}

// ============================================================================
// Device-side CTC backward + gradient for a single batch element
// ============================================================================

template <typename Type, typename IndexType>
SD_DEVICE void ctcBackwardAndGradDevice(
    Type forwardLogLoss,
    Type* alphaPtr,           // alpha workspace [lenT * lenSB]
    Type* bettaPtr,           // beta workspace [lenT * lenSB], pre-filled with negInf
    const Type* logP,         // logits pointer for this batch
    LongType incP,
    Type* gradPtr,            // gradient pointer for this batch [lenT * lenK]
    LongType incG,
    const IndexType* lbl,
    LongType elwiseP,
    LongType elwiseS,
    LongType elwiseG,
    LongType lenS,
    LongType lenT,
    LongType lenK,
    int blankIndex) {

  Type negInf = d_negative_infinity<Type>();
  LongType lenSB = 2 * lenS + 1;
  LongType incA = lenSB;

  const Type* origLogP = logP;
  Type* origAlphaPtr = alphaPtr;
  Type* origBettaPtr = bettaPtr;

  // Move to last frame
  bettaPtr += (lenT - 1) * incA;
  logP += (lenT - 1) * incP;

  // Initialize beta at t=lenT-1
  bettaPtr[lenSB - 1] = logP[blankIndex * elwiseP];
  IndexType lblIndex = lbl[(lenS - 1) * elwiseS];
  bettaPtr[lenSB - 2] = logP[lblIndex * elwiseP];

  Type* bettaPrevPtr = bettaPtr;
  bettaPtr -= incA;
  logP -= incP;

  // Backward pass
  for (LongType t = lenT - 2; t >= 0; t--) {
    LongType end = 2 * t + 2;
    if (end > lenSB - 1) end = lenSB - 1;

    for (LongType s = end; s >= 0; s--) {
      LongType ind = s / 2;
      IndexType currentInd = (s % 2 == 0) ? (IndexType)blankIndex : lbl[ind * elwiseS];

      Type bettaS = bettaPrevPtr[s];
      Type bettaS_1 = s < lenSB - 1 ? bettaPrevPtr[s + 1] : negInf;
      Type currentProb = logP[currentInd * elwiseP];

      if (s < lenSB - 2 && currentInd != (IndexType)blankIndex &&
          currentInd != lbl[(ind + 1) * elwiseS]) {
        Type bettaS_2 = bettaPrevPtr[s + 2];
        bettaPtr[s] = d_log_sum_exp3<Type>(bettaS, bettaS_1, bettaS_2) + currentProb;
      } else {
        bettaPtr[s] = d_log_sum_exp<Type>(bettaS, bettaS_1) + currentProb;
      }
    }

    bettaPrevPtr = bettaPtr;
    bettaPtr -= incA;
    logP -= incP;
  }

  // Compute gradients: alpha*beta combination
  bettaPtr = origBettaPtr;
  logP = origLogP;
  alphaPtr = origAlphaPtr;

  for (LongType t = 0; t < lenT; t++) {
    // Accumulate alpha*beta for each label index
    for (LongType s = 0; s < lenSB; s++) {
      LongType ind = s / 2;
      IndexType currentInd = (s % 2 == 0) ? (IndexType)blankIndex : lbl[ind * elwiseS];
      Type alphaBettaS = alphaPtr[s] + bettaPtr[s];

      Type& currentGrad = gradPtr[currentInd * elwiseG];
      if (currentGrad == negInf) {
        currentGrad = alphaBettaS;
      } else {
        currentGrad = d_log_sum_exp<Type>(currentGrad, alphaBettaS);
      }
    }

    // Finalize gradient for this time step
    for (LongType k = 0; k < lenK; k++) {
      Type currentProb = logP[k * elwiseP];
      Type& currentGrad = gradPtr[k * elwiseG];
      Type p2 = math::p_exp<Type>(currentGrad + forwardLogLoss - currentProb);
      currentGrad = math::p_exp<Type>(currentProb) - p2;
    }

    gradPtr += incG;
    bettaPtr += incA;
    alphaPtr += incA;
    logP += incP;
  }
}

// ============================================================================
// CUDA kernel: one thread per batch element
// ============================================================================

template <typename Type, typename IndexType>
SD_KERNEL void ctcLossKernel(
    const Type* logP,            // logits [batch, maxT, K]
    const IndexType* lblPtr,     // labels [batch, maxS]
    const IndexType* lenTPtr,    // input lengths [batch]
    const IndexType* lenSPtr,    // label lengths [batch]
    Type* logLossPtr,            // output losses [batch] (can be null)
    Type* gradPtr,               // output gradients [batch, maxT, K] (can be null)
    Type* workspace,             // alpha/beta workspace
    LongType batchP,             // logits batch stride
    LongType incP,               // logits time stride
    LongType elwiseP,            // logits class stride
    LongType batchLbl,           // labels batch stride
    LongType elwiseS,            // labels element stride
    LongType elwiseT,            // inputLengths element stride
    LongType elwiseSLen,         // labelLengths element stride
    LongType elwiseLL,           // logLoss element stride
    LongType batchG,             // gradient batch stride
    LongType incG,               // gradient time stride
    LongType elwiseG,            // gradient class stride
    LongType maxLenT,
    LongType maxLenS,
    LongType lenK,
    int blankIndex,
    LongType lenBatch,
    bool hasLoss,
    bool hasGrad,
    LongType workspacePerBatch)  // workspace size per batch element
{
  LongType batchIndex = (LongType)blockIdx.x * blockDim.x + threadIdx.x;
  if (batchIndex >= lenBatch) return;

  Type negInf = d_negative_infinity<Type>();

  // Get actual lengths for this batch element
  LongType lenT = (LongType)lenTPtr[batchIndex * elwiseT];
  LongType lenS = (LongType)lenSPtr[batchIndex * elwiseSLen];
  if (lenT > maxLenT) lenT = maxLenT;
  if (lenS > maxLenS) lenS = maxLenS;

  if (lenS <= 0 || lenT <= 0) {
    if (hasLoss) logLossPtr[batchIndex * elwiseLL] = negInf;
    return;
  }
  if (lenS > lenT) lenS = lenT;

  LongType lenSB = 2 * lenS + 1;

  // Point to this batch's logits and labels
  const Type* batchLogP = logP + batchIndex * batchP;
  const IndexType* batchLblPtr = lblPtr + batchIndex * batchLbl;

  // Point to this batch's workspace: need 2*lenT*lenSB for alpha+beta (or 1x for forward only)
  // Workspace layout: [alpha: lenT*lenSB] [beta: lenT*lenSB]
  Type* batchWorkspace = workspace + batchIndex * workspacePerBatch;
  Type* alphaWs = batchWorkspace;
  Type* betaWs = batchWorkspace + lenT * lenSB;

  // Initialize workspace with negInf
  LongType wsSize = hasGrad ? 2 * lenT * lenSB : lenT * lenSB;
  for (LongType i = 0; i < wsSize; i++) {
    batchWorkspace[i] = negInf;
  }

  // Forward pass
  Type logLoss = ctcForwardDevice<Type, IndexType>(
      alphaWs, batchLogP, incP, batchLblPtr, elwiseP, elwiseS, lenSB, lenT, blankIndex);

  if (hasLoss) {
    logLossPtr[batchIndex * elwiseLL] = logLoss;
  }

  // Backward + gradient if needed
  if (hasGrad) {
    Type* batchGradPtr = gradPtr + batchIndex * batchG;

    // Initialize gradients with negInf
    Type* tempGrad = batchGradPtr;
    for (LongType t = 0; t < lenT; t++) {
      for (LongType k = 0; k < lenK; k++) {
        tempGrad[k * elwiseG] = negInf;
      }
      tempGrad += incG;
    }

    ctcBackwardAndGradDevice<Type, IndexType>(
        logLoss, alphaWs, betaWs, batchLogP, incP,
        batchGradPtr, incG, batchLblPtr,
        elwiseP, elwiseS, elwiseG,
        lenS, lenT, lenK, blankIndex);
  }
}

// ============================================================================
// Host-side dispatch
// ============================================================================

template <typename Type, typename IndexType>
void ctc_loss_(NDArray& logits, NDArray& targetLabels, NDArray& logitsLengths,
               NDArray& targetLabelLengths, NDArray& logLosses, NDArray& gradients,
               int blankIndex) {

  auto lenBatch = logits.shapeOf()[0];
  auto maxLenT = logits.shapeOf()[1];
  auto lenK = logits.shapeOf()[2];
  auto maxLenS = targetLabels.shapeOf()[1];

  auto batchP = logits.stridesOf()[0];
  auto incP = logits.stridesOf()[1];
  auto elwiseP = logits.stridesOf()[2];

  auto batchLbl = targetLabels.stridesOf()[0];
  auto elwiseS = targetLabels.stridesOf()[1];

  auto elwiseSLen = targetLabelLengths.stridesOf()[0];
  auto elwiseT = logitsLengths.stridesOf()[0];

  // Default blankIndex if invalid
  if (blankIndex > lenK || blankIndex < 0) blankIndex = lenK - 1;

  bool hasLoss = !logLosses.isEmpty();
  bool hasGrad = !gradients.isEmpty();

  LongType elwiseLL = 0;
  if (hasLoss) elwiseLL = logLosses.stridesOf()[0];

  LongType batchG = 0, incG = 0, elwiseG = 0;
  if (hasGrad) {
    batchG = gradients.stridesOf()[0];
    incG = gradients.stridesOf()[1];
    elwiseG = gradients.stridesOf()[2];
  }

  // Workspace: per batch element, need alpha[maxT * maxSB] + beta[maxT * maxSB]
  LongType maxLenSB = 2 * maxLenS + 1;
  LongType workspacePerBatch = hasGrad ? 2 * maxLenT * maxLenSB : maxLenT * maxLenSB;

  // Allocate workspace on device
  auto context = logits.getContext();
  auto stream = context->getCudaStream();

  auto workspaceArr = NDArrayFactory::create<Type>('c', {lenBatch * workspacePerBatch});

  // Prepare arrays for special use (sync host->device as needed)
  if (hasLoss && hasGrad) {
    NDArray::prepareSpecialUse({&logLosses, &gradients}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  } else if (hasLoss) {
    NDArray::prepareSpecialUse({&logLosses}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  } else if (hasGrad) {
    NDArray::prepareSpecialUse({&gradients}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  }

  // Launch kernel: one thread per batch element
  // Use 128 threads per block as a reasonable default
  int threadsPerBlock = 128;
  int blocksPerGrid = (lenBatch + threadsPerBlock - 1) / threadsPerBlock;
  // For small batches, reduce threads per block
  if (lenBatch < threadsPerBlock) {
    threadsPerBlock = lenBatch;
    blocksPerGrid = 1;
  }

  ctcLossKernel<Type, IndexType><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(
      reinterpret_cast<const Type*>(logits.specialBuffer()),
      reinterpret_cast<const IndexType*>(targetLabels.specialBuffer()),
      reinterpret_cast<const IndexType*>(logitsLengths.specialBuffer()),
      reinterpret_cast<const IndexType*>(targetLabelLengths.specialBuffer()),
      hasLoss ? reinterpret_cast<Type*>(logLosses.specialBuffer()) : nullptr,
      hasGrad ? reinterpret_cast<Type*>(gradients.specialBuffer()) : nullptr,
      reinterpret_cast<Type*>(workspaceArr->specialBuffer()),  // workspace on device
      batchP, incP, elwiseP,
      batchLbl, elwiseS,
      elwiseT, elwiseSLen, elwiseLL,
      batchG, incG, elwiseG,
      maxLenT, maxLenS, lenK,
      blankIndex, lenBatch,
      hasLoss, hasGrad,
      workspacePerBatch);

  // Register special use (mark device buffers as updated)
  if (hasLoss && hasGrad) {
    NDArray::registerSpecialUse({&logLosses, &gradients}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  } else if (hasLoss) {
    NDArray::registerSpecialUse({&logLosses}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  } else if (hasGrad) {
    NDArray::registerSpecialUse({&gradients}, {&logits, &targetLabels, &logitsLengths, &targetLabelLengths});
  }

  delete workspaceArr;
}

// ============================================================================
// Entry point
// ============================================================================

void ctcLoss(graph::Context& block, NDArray& logitsInput, NDArray& targetLabels,
             NDArray& logitsLengths, NDArray& targetLabelLengths, NDArray& logLosses,
             NDArray& gradients, int blankIndex) {
  BUILD_DOUBLE_SELECTOR(logitsInput.dataType(), targetLabels.dataType(), ctc_loss_,
                        (logitsInput, targetLabels, logitsLengths, targetLabelLengths, logLosses, gradients, blankIndex),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}

BUILD_DOUBLE_TEMPLATE(void ctc_loss_,
                      (NDArray& logits, NDArray& targetLabels, NDArray& logitsLengths,
                       NDArray& targetLabelLengths, NDArray& logLosses, NDArray& gradients, int blankIndex),
                      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
