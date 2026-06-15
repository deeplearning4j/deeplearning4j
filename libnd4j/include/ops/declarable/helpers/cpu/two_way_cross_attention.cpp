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

// Helper: scale all elements of matrix by a scalar value
static void scaleMatrix(NDArray& matrix, double scale) {
    auto len = matrix.lengthOf();
    for (sd::LongType i = 0; i < len; i++) {
        matrix.p(i, matrix.e<double>(i) * scale);
    }
}

// Helper: compute dL/dLogits given dL/dOutput, values, and attention weights (softmax output)
static void softmaxBackward(NDArray& attnWeights, NDArray& dLdAttnWeights, NDArray& dLdLogits) {
    auto rows = attnWeights.sizeAt(0);
    auto cols = attnWeights.sizeAt(1);

    for (sd::LongType r = 0; r < rows; r++) {
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

// Process one batch slice for forward pass:
//   tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV
//   imageOut = softmax(imageQ @ tokenK^T * scale) @ tokenV
static void twoWayCrossAttentionSingle(
    NDArray* tokenQuery, NDArray* tokenKey, NDArray* tokenValue,
    NDArray* imageQuery, NDArray* imageKey, NDArray* imageValue,
    NDArray* tokenOutput, NDArray* imageOutput,
    double scale) {

    // Direction 1: token attends to image
    {
        NDArray* imageKT = imageKey->transpose();
        sd::LongType tqLen = tokenQuery->sizeAt(0);
        sd::LongType ikLen = imageKey->sizeAt(0);
        std::vector<sd::LongType> logitsShape = {tqLen, ikLen};
        NDArray logits('c', logitsShape, tokenQuery->dataType());
        MmulHelper::mmul(tokenQuery, imageKT, &logits);
        delete imageKT;
        scaleMatrix(logits, scale);
        applySoftmax2D(logits);
        MmulHelper::mmul(&logits, imageValue, tokenOutput);
    }

    // Direction 2: image attends to token
    {
        NDArray* tokenKT = tokenKey->transpose();
        sd::LongType iqLen = imageQuery->sizeAt(0);
        sd::LongType tkLen = tokenKey->sizeAt(0);
        std::vector<sd::LongType> logitsShape = {iqLen, tkLen};
        NDArray logits('c', logitsShape, imageQuery->dataType());
        MmulHelper::mmul(imageQuery, tokenKT, &logits);
        delete tokenKT;
        scaleMatrix(logits, scale);
        applySoftmax2D(logits);
        MmulHelper::mmul(&logits, tokenValue, imageOutput);
    }
}

void twoWayCrossAttention(NDArray* tokenQuery, NDArray* tokenKey,
                              NDArray* tokenValue, NDArray* imageQuery,
                              NDArray* imageKey, NDArray* imageValue,
                              NDArray* tokenOutput, NDArray* imageOutput,
                              double scale, LaunchContext* context) {
    int rank = tokenQuery->rankOf();

    if (rank == 2) {
        // 2D: [seqLen, dim]
        twoWayCrossAttentionSingle(tokenQuery, tokenKey, tokenValue,
                                   imageQuery, imageKey, imageValue,
                                   tokenOutput, imageOutput, scale);
    } else {
        // 3D: [batch, seqLen, dim] — iterate over batch
        sd::LongType batchSize = tokenQuery->sizeAt(0);
        std::vector<sd::LongType> dimsToExclude = {0};
        for (sd::LongType b = 0; b < batchSize; b++) {
            NDArray tqSlice = *(*tokenQuery)(b, dimsToExclude);
            NDArray tkSlice = *(*tokenKey)(b, dimsToExclude);
            NDArray tvSlice = *(*tokenValue)(b, dimsToExclude);
            NDArray iqSlice = *(*imageQuery)(b, dimsToExclude);
            NDArray ikSlice = *(*imageKey)(b, dimsToExclude);
            NDArray ivSlice = *(*imageValue)(b, dimsToExclude);
            NDArray toSlice = *(*tokenOutput)(b, dimsToExclude);
            NDArray ioSlice = *(*imageOutput)(b, dimsToExclude);

            twoWayCrossAttentionSingle(&tqSlice, &tkSlice, &tvSlice,
                                       &iqSlice, &ikSlice, &ivSlice,
                                       &toSlice, &ioSlice, scale);
        }
    }
}

// Process one batch slice for backward pass
static void twoWayCrossAttentionBpSingle(
    NDArray* tokenQuery, NDArray* tokenKey, NDArray* tokenValue,
    NDArray* imageQuery, NDArray* imageKey, NDArray* imageValue,
    NDArray* gradTokenOut, NDArray* gradImageOut,
    NDArray* dLdTokenQuery, NDArray* dLdTokenKey, NDArray* dLdTokenValue,
    NDArray* dLdImageQuery, NDArray* dLdImageKey, NDArray* dLdImageValue,
    double scale, LaunchContext* context) {

    // ---- Direction 1 backward: tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV ----
    {
        NDArray* imageKT = imageKey->transpose();
        sd::LongType tqLen = tokenQuery->sizeAt(0);
        sd::LongType ikLen = imageKey->sizeAt(0);
        std::vector<sd::LongType> logitsShape = {tqLen, ikLen};
        NDArray logits1('c', logitsShape, tokenQuery->dataType());
        MmulHelper::mmul(tokenQuery, imageKT, &logits1);
        delete imageKT;
        scaleMatrix(logits1, scale);

        NDArray attnWeights1(logits1.shapeInfo(), false, context, true);
        attnWeights1.assign(&logits1);
        applySoftmax2D(attnWeights1);

        // dL/dAttnWeights1 = gradTokenOut @ imageV^T
        NDArray* imageVT = imageValue->transpose();
        std::vector<sd::LongType> attnShape = {tqLen, ikLen};
        NDArray dLdAttn1('c', attnShape, tokenQuery->dataType());
        MmulHelper::mmul(gradTokenOut, imageVT, &dLdAttn1);
        delete imageVT;

        // dL/dImageValue += attnWeights1^T @ gradTokenOut
        NDArray* attnWeights1T = attnWeights1.transpose();
        MmulHelper::mmul(attnWeights1T, gradTokenOut, dLdImageValue);
        delete attnWeights1T;

        // dL/dLogits1 through softmax backward
        NDArray dLdLogits1(dLdAttn1.shapeInfo(), false, context, true);
        softmaxBackward(attnWeights1, dLdAttn1, dLdLogits1);
        scaleMatrix(dLdLogits1, scale);

        // dL/dTokenQuery += dLdLogits1 @ imageKey
        MmulHelper::mmul(&dLdLogits1, imageKey, dLdTokenQuery);

        // dL/dImageKey += dLdLogits1^T @ tokenQuery
        NDArray* dLdLogits1T = dLdLogits1.transpose();
        std::vector<sd::LongType> ikShape = {imageKey->sizeAt(0), imageKey->sizeAt(1)};
        NDArray dLdImageKeyContrib('c', ikShape, imageKey->dataType());
        MmulHelper::mmul(dLdLogits1T, tokenQuery, &dLdImageKeyContrib);
        delete dLdLogits1T;
        dLdImageKey->assign(&dLdImageKeyContrib);
    }

    // ---- Direction 2 backward: imageOut = softmax(imageQ @ tokenK^T * scale) @ tokenV ----
    {
        NDArray* tokenKT = tokenKey->transpose();
        sd::LongType iqLen = imageQuery->sizeAt(0);
        sd::LongType tkLen = tokenKey->sizeAt(0);
        std::vector<sd::LongType> logitsShape = {iqLen, tkLen};
        NDArray logits2('c', logitsShape, imageQuery->dataType());
        MmulHelper::mmul(imageQuery, tokenKT, &logits2);
        delete tokenKT;
        scaleMatrix(logits2, scale);

        NDArray attnWeights2(logits2.shapeInfo(), false, context, true);
        attnWeights2.assign(&logits2);
        applySoftmax2D(attnWeights2);

        // dL/dAttnWeights2 = gradImageOut @ tokenV^T
        NDArray* tokenVT = tokenValue->transpose();
        std::vector<sd::LongType> attnShape = {iqLen, tkLen};
        NDArray dLdAttn2('c', attnShape, imageQuery->dataType());
        MmulHelper::mmul(gradImageOut, tokenVT, &dLdAttn2);
        delete tokenVT;

        // dL/dTokenValue += attnWeights2^T @ gradImageOut
        NDArray* attnWeights2T = attnWeights2.transpose();
        MmulHelper::mmul(attnWeights2T, gradImageOut, dLdTokenValue);
        delete attnWeights2T;

        // dL/dLogits2 through softmax backward
        NDArray dLdLogits2(dLdAttn2.shapeInfo(), false, context, true);
        softmaxBackward(attnWeights2, dLdAttn2, dLdLogits2);
        scaleMatrix(dLdLogits2, scale);

        // dL/dImageQuery += dLdLogits2 @ tokenKey
        MmulHelper::mmul(&dLdLogits2, tokenKey, dLdImageQuery);

        // dL/dTokenKey += dLdLogits2^T @ imageQuery
        NDArray* dLdLogits2T = dLdLogits2.transpose();
        std::vector<sd::LongType> tkShape = {tokenKey->sizeAt(0), tokenKey->sizeAt(1)};
        NDArray dLdTokenKeyContrib('c', tkShape, tokenKey->dataType());
        MmulHelper::mmul(dLdLogits2T, imageQuery, &dLdTokenKeyContrib);
        delete dLdLogits2T;
        dLdTokenKey->assign(&dLdTokenKeyContrib);
    }
}

void twoWayCrossAttentionBp(
    NDArray* tokenQuery, NDArray* tokenKey, NDArray* tokenValue,
    NDArray* imageQuery, NDArray* imageKey, NDArray* imageValue,
    NDArray* gradTokenOut, NDArray* gradImageOut,
    NDArray* dLdTokenQuery, NDArray* dLdTokenKey, NDArray* dLdTokenValue,
    NDArray* dLdImageQuery, NDArray* dLdImageKey, NDArray* dLdImageValue,
    double scale, LaunchContext* context) {

    int rank = tokenQuery->rankOf();

    if (rank == 2) {
        // 2D: [seqLen, dim]
        twoWayCrossAttentionBpSingle(
            tokenQuery, tokenKey, tokenValue,
            imageQuery, imageKey, imageValue,
            gradTokenOut, gradImageOut,
            dLdTokenQuery, dLdTokenKey, dLdTokenValue,
            dLdImageQuery, dLdImageKey, dLdImageValue,
            scale, context);
    } else {
        // 3D: [batch, seqLen, dim] — iterate over batch
        sd::LongType batchSize = tokenQuery->sizeAt(0);

        // Zero-initialize all gradient outputs before accumulation
        dLdTokenQuery->nullify();
        dLdTokenKey->nullify();
        dLdTokenValue->nullify();
        dLdImageQuery->nullify();
        dLdImageKey->nullify();
        dLdImageValue->nullify();

        std::vector<sd::LongType> dimsToExclude = {0};
        for (sd::LongType b = 0; b < batchSize; b++) {
            NDArray tqSlice  = *(*tokenQuery)(b, dimsToExclude);
            NDArray tkSlice  = *(*tokenKey)(b, dimsToExclude);
            NDArray tvSlice  = *(*tokenValue)(b, dimsToExclude);
            NDArray iqSlice  = *(*imageQuery)(b, dimsToExclude);
            NDArray ikSlice  = *(*imageKey)(b, dimsToExclude);
            NDArray ivSlice  = *(*imageValue)(b, dimsToExclude);
            NDArray gtSlice  = *(*gradTokenOut)(b, dimsToExclude);
            NDArray giSlice  = *(*gradImageOut)(b, dimsToExclude);

            // Allocate per-batch gradient slices
            NDArray dTQ(tqSlice.shapeInfo(), false, context, true);
            NDArray dTK(tkSlice.shapeInfo(), false, context, true);
            NDArray dTV(tvSlice.shapeInfo(), false, context, true);
            NDArray dIQ(iqSlice.shapeInfo(), false, context, true);
            NDArray dIK(ikSlice.shapeInfo(), false, context, true);
            NDArray dIV(ivSlice.shapeInfo(), false, context, true);

            twoWayCrossAttentionBpSingle(
                &tqSlice, &tkSlice, &tvSlice,
                &iqSlice, &ikSlice, &ivSlice,
                &gtSlice, &giSlice,
                &dTQ, &dTK, &dTV, &dIQ, &dIK, &dIV,
                scale, context);

            // Write per-batch results into output slices
            NDArray dLdTQSlice = *(*dLdTokenQuery)(b, dimsToExclude);
            NDArray dLdTKSlice = *(*dLdTokenKey)(b, dimsToExclude);
            NDArray dLdTVSlice = *(*dLdTokenValue)(b, dimsToExclude);
            NDArray dLdIQSlice = *(*dLdImageQuery)(b, dimsToExclude);
            NDArray dLdIKSlice = *(*dLdImageKey)(b, dimsToExclude);
            NDArray dLdIVSlice = *(*dLdImageValue)(b, dimsToExclude);

            dLdTQSlice.assign(&dTQ);
            dLdTKSlice.assign(&dTK);
            dLdTVSlice.assign(&dTV);
            dLdIQSlice.assign(&dIQ);
            dLdIKSlice.assign(&dIK);
            dLdIVSlice.assign(&dIV);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
