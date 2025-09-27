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
// @author AbdelRauf
//
#include <execution/ThreadPool.h>
#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/ctc.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <system/selective_rendering.h>
#if NOT_EXCLUDED(OP_ctc_loss)
namespace sd {
namespace ops {
namespace helpers {

template <bool IsLogPStrided = false, bool IsLblStrided = false, typename Type, typename IndexType>
Type forward(Type *alphaPtr, const sd::LongType &incA, const Type *logP, const sd::LongType &incP, const IndexType *lbl,
             const sd::LongType &lenSB, const sd::LongType &lenT, const int &blankIndex, int elwiseP = 1,
             int elwiseS = 1) {
  Type negInf = negative_infinity<Type>();
  // initialize alphas at t=0
  alphaPtr[0] = element<IsLogPStrided>(logP, blankIndex, elwiseP);
  // alphaPtr[1] =logP[lbl[0]];
  alphaPtr[1] = element<IsLogPStrided>(logP, *lbl, elwiseP);
  // the rest initialization was skipped
  // as its assumed the array already were initialized with negative infinity
  // move to the next frame
  Type *alphaPrevPtr = alphaPtr;
  alphaPtr += incA;
  logP += incP;

  auto startX = lenSB - 2 * lenT;
  // process the rest
  for (auto t = 1; t < lenT; t++) {
    // start = max(0,L-2*(T-t))
    auto s = startX + 2 * t;
    s = s > 0 ? s : 0;
    for (; s < lenSB; s++) {
      auto ind = s / 2;  // our real index
      // we force blanks for even indexes
      // strided version of lbl[ind] => element<IsLblStrided>(lbl, ind, elwiseS)
      auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);
      // {t-1,s}
      Type alphaS = alphaPrevPtr[s];
      Type alphaS_1 = s > 0 ? alphaPrevPtr[s - 1] : negInf;
      // logP[currentInd] or logP[currentInd*elwiseP]
      auto currentProb = element<IsLogPStrided>(logP, currentInd, elwiseP);
      // if blank or the same as previous
      if (s > 1 && currentInd != blankIndex && currentInd != element<IsLblStrided>(lbl, ind - 1, elwiseS)) {
        Type alphaS_2 = alphaPrevPtr[s - 2];
        alphaPtr[s] = log_sum_exp(alphaS, alphaS_1, alphaS_2) + currentProb;
      } else {
        alphaPtr[s] = log_sum_exp(alphaS, alphaS_1) + currentProb;
      }
    }

    // store t-1 alpha Ptr
    alphaPrevPtr = alphaPtr;
    logP += incP;
    alphaPtr += incA;
  }
  auto logP0 = alphaPrevPtr[lenSB - 1];
  auto logP1 = alphaPrevPtr[lenSB - 2];
  return -log_sum_exp(logP0, logP1);
}

//#undef CALCULATE_ALL_IN_ONE_FRAME_LOOP

template <bool IsLogPStrided = false, bool IsLblStrided = false, bool isGradStrided = false, typename Type,
          typename IndexType = int>
void backwardAndGrad(Type forwardLogLoss, Type *alphaPtr, Type *bettaPtr, int incA, const Type *logP, int incP,
                     Type *gradPtr, int incG, const IndexType *lbl, const sd::LongType &lenS, const sd::LongType &lenT,
                     const sd::LongType &lenK, const int &blankIndex, int elwiseP = 1, int elwiseS = 1,
                     int elwiseG = 1) {
  Type negInf = negative_infinity<Type>();
  sd::LongType lenSB = 2 * lenS + 1;
  auto origBetta = bettaPtr;
  auto origLogP = logP;
  // move to the last frame
  bettaPtr += (lenT - 1) * incA;
  logP += (lenT - 1) * incP;

  // initialize bettas at t=lenT
  bettaPtr[lenSB - 1] = element<IsLogPStrided>(logP, blankIndex, elwiseP);
  auto lblIndex = element<IsLblStrided>(lbl, lenS - 1, elwiseS);
  bettaPtr[lenSB - 2] = element<IsLogPStrided>(logP, lblIndex, elwiseP);  // logP[lbl[lenS - 1]];

#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
  // move to the last
  gradPtr += (lenT - 1) * incG;
  alphaPtr += (lenT - 1) * incA;
  for (auto s = lenSB - 1; s >= 0; s--) {
    auto ind = s / 2;  // our real index
                       // we forced blanks for even indexes
    auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);
    // alpha(s)*betta(s) in log scale but still store in alpha to save memory
    auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

    // sum  (alpha(s)*betta(s) ) over real indexes
    auto &currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG);  // gradPtr[currentInd];
    if (currentGrad == negInf) {
      currentGrad = alphaBettaS;
    } else {
      Type cMax = std::max(currentGrad, alphaBettaS);
      currentGrad = std::log(std::exp(currentGrad - cMax) + std::exp(alphaBettaS - cMax)) + cMax;
    }
  }
  for (int k = 0; k < lenK; k++) {
    // compute the rest grad

    // prob(t,k) - grad(k) / ((prob(t,k)*Z) )

    // p2= grad(k) / (prob(t,k)*Z )
    // in logscale . plus we have Z as -logLoss
    // auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
    // gradPtr[k] = std::exp(logP[k]) - p2;
    auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
    auto &currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
    auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
    currentGrad = std::exp(currentProb) - p2;
  }
  gradPtr -= incG;
  alphaPtr -= incA;
#endif

  auto bettaPrevPtr = bettaPtr;
  bettaPtr -= incA;
  logP -= incP;
  // process the rest
  for (auto t = lenT - 2; t >= 0; t--) {
#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
    auto end = lenSB - 1;
#else
    auto end = std::min(2 * t + 2, lenSB - 1);
#endif
    for (auto s = end; s >= 0; s--) {
      auto ind = s / 2;  // our real index
      // we forced blanks for even indexes
      auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);  // lbl[ind];
      // {t-1,s}
      Type bettaS = bettaPrevPtr[s];
      Type bettaS_1 = s < lenSB - 1 ? bettaPrevPtr[s + 1] : negInf;
      // logP[currentInd]
      auto currentProb = element<IsLogPStrided>(logP, currentInd, elwiseP);
      // if blank or the same as previous
      if (s < lenSB - 2 && currentInd != blankIndex && currentInd != element<IsLblStrided>(lbl, ind + 1, elwiseS)) {
        Type bettaS_2 = bettaPrevPtr[s + 2];
        bettaPtr[s] = log_sum_exp(bettaS, bettaS_1, bettaS_2) + currentProb;
      } else {
        bettaPtr[s] = log_sum_exp(bettaS, bettaS_1) + currentProb;
      }

#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
      // alpha(s)*betta(s) in log scale but still store in alpha to save memory
      auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

      // sum  (alpha(s)*betta(s) ) over real indexes
      auto &currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG);  // gradPtr[currentInd];
      if (currentGrad == negInf) {
        currentGrad = alphaBettaS;
      } else {
        Type cMax = std::max(currentGrad, alphaBettaS);
        currentGrad = std::log(std::exp(currentGrad - cMax) + std::exp(alphaBettaS - cMax)) + cMax;
      }

#endif
    }

#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
    for (int k = 0; k < lenK; k++) {
      // compute the rest grad

      // prob(t,k) - grad(k) / ((prob(t,k)*Z) )

      // p2= grad(k) / (prob(t,k)*Z )
      // in logscale . plus we have Z as -logLoss
      // auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
      // gradPtr[k] = std::exp(logP[k]) - p2;
      auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
      auto &currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
      auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
      currentGrad = std::exp(currentProb) - p2;
    }
    alphaPtr -= incA;
    gradPtr -= incG;
#endif

    bettaPrevPtr = bettaPtr;
    bettaPtr -= incA;
    logP -= incP;
  }


#if !defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
  // alpha*betta
  bettaPtr = origBetta;
  logP = origLogP;

  for (int t = 0; t < lenT; t++) {
    for (int s = 0; s < lenSB; s++) {
      auto ind = s / 2;  // our real index
                         // we forced blanks for even indexes
      auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);  // lbl[ind];
      // alpha(s)*betta(s) in log scale but still store in alpha to save memory
      auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

      // sum  (alpha(s)*betta(s) ) over real indexes
      auto &currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG);  // gradPtr[currentInd];
      if (currentGrad == negInf) {
        currentGrad = alphaBettaS;
      } else {
        currentGrad = log_sum_exp(currentGrad, alphaBettaS);
      }
      // alphaPtr[s] = alphaBettaS;
    }

    PRAGMA_OMP_SIMD
    for (int k = 0; k < lenK; k++) {
      // compute the rest grad

      // prob(t,k) - grad(k) / ((prob(t,k)*Z) )

      // p2= grad(k) / (prob(t,k)*Z )
      // in logscale . plus we have Z as -logLoss
      // auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
      // gradPtr[k] = std::exp(logP[k]) - p2;
      auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
      auto &currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
      auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
      currentGrad = std::exp(currentProb) - p2;
    }

    gradPtr += incG;
    bettaPtr += incA;
    alphaPtr += incA;
    logP += incP;
  }
#endif
}

/**
 * Calculates ctc loss and fills gradients
 * @param logP logits matrix(lenT,lenK) pointer (log soft max input of rnn)
 * @param incP stride of logits for the next time frame
 * @param gradPtr gradient for output
 * @param incG  stride of the gradient for the next time frame
 * @param lbl target label
 * @param lenT frame length
 * @param lenK class length
 * @param lenS target label length
 * @param blankIndex index of the blank label in logit class
 */
template <bool IsLogPStrided = true, bool IsLblStrided = true, bool IsGradStrided = true, typename Type,
          typename IndexType>
Type unitLossAndGrad(const Type *logP, int incP, Type *gradPtr, int incG, const IndexType *lbl, int lenT, int lenK,
                     int lenS, int blankIndex, int elwiseP = 1, int elwiseS = 1, int elwiseG = 1) {
  auto lenSB = 2 * lenS + 1;
  // create temp Array for holding bettaArr [lenT,lenSB]
  // create temp Array for holding alphaArr [lenT,lenSB]
  int bufferC = gradPtr ? 2 : 1;
  NDArray bufferArr = NDArrayFactory::create<Type>('c', {bufferC, lenT, lenSB});
  auto bufferPtr = bufferArr.bufferAsT<Type>();
  auto incA = bufferArr.stridesOf()[1];
  auto bettaBufferPtr = bufferPtr + bufferArr.stridesOf()[0];
  Type negInf = negative_infinity<Type>();

#if 1
  if (gradPtr) {
    if (elwiseG == 1) {
      PRAGMA_OMP_SIMD
      for (int i = 0; i < lenK * lenT; i++) {
        gradPtr[i] = negInf;
      }
    } else {
      auto tempPtr = gradPtr;
      for (int i = 0; i < lenT; i++) {
        for (int j = 0; j < lenK; j++) element<false>(tempPtr, j, elwiseG) = negInf;
        tempPtr += incG;
      }
    }
  }
#endif

  // set all vals to neginf
  PRAGMA_OMP_SIMD
  for (int i = 0; i < bufferC * lenSB * lenT; i++) {
    bufferPtr[i] = negInf;
  }

  // forward
  Type logLoss =
      forward<IsLogPStrided, IsLblStrided>(bufferPtr, incA, logP, incP, lbl, lenSB, lenT, blankIndex, elwiseP, elwiseS);
  // backward and gradient if gradptr supplied
  if (gradPtr)
    backwardAndGrad<IsLogPStrided, IsLblStrided, IsGradStrided>(logLoss, bufferPtr, bettaBufferPtr, incA, logP, incP,
                                                                gradPtr, incG, lbl, lenS, lenT, lenK, blankIndex,
                                                                elwiseP, elwiseS, elwiseG);
  return logLoss;
}

template <typename Type, typename IndexType>
void ctc_loss_(NDArray&logits, NDArray&targetLabels, NDArray&logitsLengths,
               NDArray&targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex) {
  // lenT  - input length of T
  // lenS  - lenght of sequence
  // lenSB - length with blanks
  auto lenBatch = logits.shapeOf()[0];

  auto maxLenT = logits.shapeOf()[1];
  auto lenK = logits.shapeOf()[2];
  auto maxLenS = targetLabels.shapeOf()[1];

  // get probability buffer and targetLabels buffer
  auto logP = logits.bufferAsT<Type>();
  auto lblPtr = targetLabels.bufferAsT<IndexType>();

  auto lenTPtr = logitsLengths.bufferAsT<IndexType>();
  auto lenSPtr = targetLabelLengths.bufferAsT<IndexType>();

  auto batchLbl = targetLabels.stridesOf()[0];
  auto batchP = logits.stridesOf()[0];
  auto incP = logits.stridesOf()[1];

  auto elwiseSLen = targetLabelLengths.stridesOf()[0];
  auto elwiseT = logitsLengths.stridesOf()[0];
  auto elwiseS = targetLabels.stridesOf()[1];
  auto elwiseP = logits.stridesOf()[2];

  int elwiseLL = 0;
  Type *logLossPtr = nullptr;
  if (!logLosses.isEmpty()) {
    elwiseLL = logLosses.stridesOf()[0];
    logLossPtr = logLosses.bufferAsT<Type>();
  }
  // defaulting blankIndex to the last class if its incorrect or -1
  if (blankIndex > maxLenS || blankIndex < 0) blankIndex = maxLenS - 1;
  auto func = [logP, batchP, incP, elwiseP, lenK, lenTPtr, lenSPtr, logLossPtr, lblPtr, maxLenT, maxLenS, batchLbl,
               blankIndex, elwiseT, elwiseLL, elwiseSLen, elwiseS,
               &gradients](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
    Type *gradPtr = nullptr;
    Type resultLoss;
    int batchG, incG, elwiseG;
    if (!gradients.isEmpty()) {
      batchG = gradients.stridesOf()[0];
      incG = gradients.stridesOf()[1];
      elwiseG = gradients.stridesOf()[2];
      gradPtr = gradients.bufferAsT<Type>() + start * batchG;
    } else {
      elwiseG = 1;
    }
    auto logPtr = logP + start * batchP;
    auto tempLblPtr = lblPtr + start * batchLbl;

    if (elwiseP == 1 && elwiseS == 1 && elwiseG == 1) {
      // choose ews one
      for (int batchIndex = start; batchIndex < stop; batchIndex += increment) {
        auto lenT = lenTPtr[batchIndex * elwiseT];
        auto lenS = lenSPtr[batchIndex * elwiseSLen];
        lenT = lenT > maxLenT ? maxLenT : lenT;
        lenS = lenS > maxLenS ? maxLenS : lenS;
        if (lenS <= 0 || lenT <= 0) {
          resultLoss = negative_infinity<Type>();
        } else {
          if (lenS > lenT) lenS = lenT;
          resultLoss = unitLossAndGrad<false, false, false, Type, IndexType>(logPtr, incP, gradPtr, incG, tempLblPtr,
                                                                             lenT, lenK, lenS, blankIndex);
        }
        if (gradPtr) gradPtr += batchG;
        if (logLossPtr) logLossPtr[batchIndex * elwiseLL] = resultLoss;
        logPtr += batchP;
        tempLblPtr += batchLbl;
      }
    } else {
      // slow strided case for all 3
      for (int batchIndex = start; batchIndex < stop; batchIndex += increment) {
        auto lenT = lenTPtr[batchIndex * elwiseT];
        auto lenS = lenSPtr[batchIndex * elwiseSLen];
        lenT = lenT > maxLenT ? maxLenT : lenT;
        lenS = lenS > maxLenS ? maxLenS : lenS;
        if (lenS <= 0 || lenT <= 0) {
          resultLoss = negative_infinity<Type>();
        } else {
          if (lenS > lenT) lenS = lenT;
          resultLoss = unitLossAndGrad<true, true, true, Type, IndexType>(
              logPtr, incP, gradPtr, incG, tempLblPtr, lenT, lenK, lenS, blankIndex, elwiseP, elwiseS, elwiseG);
        }
        if (gradPtr) gradPtr += batchG;
        if (logLossPtr) logLossPtr[batchIndex * elwiseLL] = resultLoss;
        logPtr += batchP;
        tempLblPtr += batchLbl;
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, lenBatch, 1);
}

void ctcLoss(graph::Context &block, NDArray&logits, NDArray&targetLabels, NDArray&logitsLengths,
             NDArray&targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex) {
  auto logitsDType = logits.dataType();
  auto targetLabelsDType = targetLabels.dataType();
  BUILD_DOUBLE_SELECTOR(logits.dataType(), targetLabels.dataType(), ctc_loss_,
                        (logits, targetLabels, logitsLengths, targetLabelLengths, logLosses, gradients, blankIndex),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}

BUILD_DOUBLE_TEMPLATE( void ctc_loss_,
                      (NDArray&logits, NDArray&targetLabels, NDArray&logitsLengths,
                       NDArray&targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex),
                      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif