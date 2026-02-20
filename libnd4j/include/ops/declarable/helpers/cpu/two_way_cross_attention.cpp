/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/two_way_cross_attention.h>
#include <helpers/MmulHelper.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

// Helper: apply row-wise softmax in-place on a 2D NDArray
static void applySoftmax2D(NDArray& matrix) {
    auto rows = matrix.sizeAt(0);
    auto cols = matrix.sizeAt(1);

    for (sd::LongType r = 0; r < rows; r++) {
        double maxVal = -1e30;
        for (sd::LongType c = 0; c < cols; c++) {
            double v = matrix.e<double>(r, c);
            if (v > maxVal) maxVal = v;
        }
        double sum = 0.0;
        for (sd::LongType c = 0; c < cols; c++) {
            double v = std::exp(matrix.e<double>(r, c) - maxVal);
            matrix.p(r, c, v);
            sum += v;
        }
        for (sd::LongType c = 0; c < cols; c++) {
            matrix.p(r, c, matrix.e<double>(r, c) / sum);
        }
    }
}

// Helper: compute dL/dLogits given dL/dOutput, values, and attention weights (softmax output)
// dL/dLogits[i,j] = sum_k(dLdOut[i,k] * V[j,k]) ... but also need softmax backward
// Full: dL/d(pre-softmax logits) where attnWeights = softmax(logits)
// dL/dLogits = attnWeights * (dL/dAttnWeights - rowwise_sum(dL/dAttnWeights * attnWeights))
// where dL/dAttnWeights = dL/dOutput @ V^T
static void softmaxBackward(NDArray& attnWeights, NDArray& dLdAttnWeights, NDArray& dLdLogits) {
    auto rows = attnWeights.sizeAt(0);
    auto cols = attnWeights.sizeAt(1);

    for (sd::LongType r = 0; r < rows; r++) {
        // Compute dot product: sum(dLdAttnWeights[r,:] * attnWeights[r,:])
        double dot = 0.0;
        for (sd::LongType c = 0; c < cols; c++) {
            dot += dLdAttnWeights.e<double>(r, c) * attnWeights.e<double>(r, c);
        }
        for (sd::LongType c = 0; c < cols; c++) {
            double w = attnWeights.e<double>(r, c);
            double dw = dLdAttnWeights.e<double>(r, c);
            dLdLogits.p(r, c, w * (dw - dot));
        }
    }
}

void twoWayCrossAttention(NDArray* tokenQuery, NDArray* tokenKey,
                              NDArray* tokenValue, NDArray* imageQuery,
                              NDArray* imageKey, NDArray* imageValue,
                              NDArray* tokenOutput, NDArray* imageOutput,
                              double scale, LaunchContext* context) {
    // Two-way cross attention:
    // Direction 1 (token-to-image): tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV
    // Direction 2 (image-to-token): imageOut = softmax(imageQ @ tokenK^T * scale) @ tokenV
    //
    // All inputs are 2D: [seqLen, dim]

    // Direction 1: token attends to image
    {
        auto imageKT = imageKey->transpose();
        NDArray logits('c', {tokenQuery->sizeAt(0), imageKey->sizeAt(0)},
                       tokenQuery->dataType(), context);
        MmulHelper::mmul(tokenQuery, &imageKT, &logits);
        logits *= scale;
        applySoftmax2D(logits);
        MmulHelper::mmul(&logits, imageValue, tokenOutput);
    }

    // Direction 2: image attends to token
    {
        auto tokenKT = tokenKey->transpose();
        NDArray logits('c', {imageQuery->sizeAt(0), tokenKey->sizeAt(0)},
                       imageQuery->dataType(), context);
        MmulHelper::mmul(imageQuery, &tokenKT, &logits);
        logits *= scale;
        applySoftmax2D(logits);
        MmulHelper::mmul(&logits, tokenValue, imageOutput);
    }
}

void twoWayCrossAttentionBp(
    NDArray* tokenQuery, NDArray* tokenKey, NDArray* tokenValue,
    NDArray* imageQuery, NDArray* imageKey, NDArray* imageValue,
    NDArray* gradTokenOut, NDArray* gradImageOut,
    NDArray* dLdTokenQuery, NDArray* dLdTokenKey, NDArray* dLdTokenValue,
    NDArray* dLdImageQuery, NDArray* dLdImageKey, NDArray* dLdImageValue,
    double scale, LaunchContext* context) {

    // ---- Direction 1 backward: tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV ----
    {
        // Recompute forward
        auto imageKT = imageKey->transpose();
        NDArray logits1('c', {tokenQuery->sizeAt(0), imageKey->sizeAt(0)},
                        tokenQuery->dataType(), context);
        MmulHelper::mmul(tokenQuery, &imageKT, &logits1);
        logits1 *= scale;

        NDArray attnWeights1(logits1.shapeInfo(), logits1.dataType(), false, context);
        attnWeights1.assign(logits1);
        applySoftmax2D(attnWeights1);

        // dL/dAttnWeights1 = gradTokenOut @ imageV^T
        auto imageVT = imageValue->transpose();
        NDArray dLdAttn1('c', {tokenQuery->sizeAt(0), imageKey->sizeAt(0)},
                         tokenQuery->dataType(), context);
        MmulHelper::mmul(gradTokenOut, &imageVT, &dLdAttn1);

        // dL/dImageValue += attnWeights1^T @ gradTokenOut
        auto attnWeights1T = attnWeights1.transpose();
        MmulHelper::mmul(&attnWeights1T, gradTokenOut, dLdImageValue);

        // dL/dLogits1 through softmax backward
        NDArray dLdLogits1(dLdAttn1.shapeInfo(), dLdAttn1.dataType(), false, context);
        softmaxBackward(attnWeights1, dLdAttn1, dLdLogits1);
        dLdLogits1 *= scale;

        // dL/dTokenQuery += dLdLogits1 @ imageKey
        MmulHelper::mmul(&dLdLogits1, imageKey, dLdTokenQuery);

        // dL/dImageKey += dLdLogits1^T @ tokenQuery
        auto dLdLogits1T = dLdLogits1.transpose();
        NDArray dLdImageKeyContrib('c', {imageKey->sizeAt(0), imageKey->sizeAt(1)},
                                    imageKey->dataType(), context);
        MmulHelper::mmul(&dLdLogits1T, tokenQuery, &dLdImageKeyContrib);
        dLdImageKey->assign(dLdImageKeyContrib);
    }

    // ---- Direction 2 backward: imageOut = softmax(imageQ @ tokenK^T * scale) @ tokenV ----
    {
        // Recompute forward
        auto tokenKT = tokenKey->transpose();
        NDArray logits2('c', {imageQuery->sizeAt(0), tokenKey->sizeAt(0)},
                        imageQuery->dataType(), context);
        MmulHelper::mmul(imageQuery, &tokenKT, &logits2);
        logits2 *= scale;

        NDArray attnWeights2(logits2.shapeInfo(), logits2.dataType(), false, context);
        attnWeights2.assign(logits2);
        applySoftmax2D(attnWeights2);

        // dL/dAttnWeights2 = gradImageOut @ tokenV^T
        auto tokenVT = tokenValue->transpose();
        NDArray dLdAttn2('c', {imageQuery->sizeAt(0), tokenKey->sizeAt(0)},
                         imageQuery->dataType(), context);
        MmulHelper::mmul(gradImageOut, &tokenVT, &dLdAttn2);

        // dL/dTokenValue += attnWeights2^T @ gradImageOut
        auto attnWeights2T = attnWeights2.transpose();
        MmulHelper::mmul(&attnWeights2T, gradImageOut, dLdTokenValue);

        // dL/dLogits2 through softmax backward
        NDArray dLdLogits2(dLdAttn2.shapeInfo(), dLdAttn2.dataType(), false, context);
        softmaxBackward(attnWeights2, dLdAttn2, dLdLogits2);
        dLdLogits2 *= scale;

        // dL/dImageQuery += dLdLogits2 @ tokenKey
        MmulHelper::mmul(&dLdLogits2, tokenKey, dLdImageQuery);

        // dL/dTokenKey += dLdLogits2^T @ imageQuery
        auto dLdLogits2T = dLdLogits2.transpose();
        NDArray dLdTokenKeyContrib('c', {tokenKey->sizeAt(0), tokenKey->sizeAt(1)},
                                    tokenKey->dataType(), context);
        MmulHelper::mmul(&dLdLogits2T, imageQuery, &dLdTokenKeyContrib);
        dLdTokenKey->assign(dLdTokenKeyContrib);
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
