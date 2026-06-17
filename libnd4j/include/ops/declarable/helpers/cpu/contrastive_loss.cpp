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

#include <ops/declarable/helpers/contrastive_loss.h>
#include <helpers/MmulHelper.h>
#include <math/templatemath.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

// Helper: compute softmax of a vector in-place (row of logits matrix)
static void rowSoftmax(double* data, sd::LongType len) {
    double maxVal = data[0];
    for (sd::LongType i = 1; i < len; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    double sum = 0.0;
    for (sd::LongType i = 0; i < len; i++) {
        data[i] = sd::math::sd_exp<double>(data[i] - maxVal);
        sum += data[i];
    }
    for (sd::LongType i = 0; i < len; i++) {
        data[i] /= sum;
    }
}

void contrastiveLoss(NDArray* imageEmbeddings, NDArray* textEmbeddings,
                        NDArray* output, double temperature,
                        LaunchContext* context) {
    auto batch = imageEmbeddings->sizeAt(0);
    auto embedDim = imageEmbeddings->sizeAt(1);

    // Compute similarity matrix: logits = image @ text^T
    NDArray* textT = textEmbeddings->transpose();
    std::vector<sd::LongType> logitsShape = {batch, batch};
    NDArray logits('c', logitsShape, imageEmbeddings->dataType());
    MmulHelper::mmul(imageEmbeddings, textT, &logits);
    delete textT;

    // Scale by temperature
    for (sd::LongType i = 0; i < batch; i++) {
        for (sd::LongType j = 0; j < batch; j++) {
            logits.p(i, j, logits.e<double>(i, j) * temperature);
        }
    }

    // Compute cross-entropy loss where target for row i and col i is index i
    double totalLoss = 0.0;

    // CE along rows (image-to-text): for each row, target = diagonal index
    for (sd::LongType i = 0; i < batch; i++) {
        std::vector<double> rowData(batch);
        for (sd::LongType j = 0; j < batch; j++) {
            rowData[j] = logits.e<double>(i, j);
        }
        rowSoftmax(rowData.data(), batch);
        double prob = std::max(rowData[i], 1e-12);
        totalLoss += -sd::math::sd_log<double>(prob);
    }

    // CE along columns (text-to-image): for each col, target = diagonal index
    for (sd::LongType j = 0; j < batch; j++) {
        std::vector<double> colData(batch);
        for (sd::LongType i = 0; i < batch; i++) {
            colData[i] = logits.e<double>(i, j);
        }
        rowSoftmax(colData.data(), batch);
        double prob = std::max(colData[j], 1e-12);
        totalLoss += -sd::math::sd_log<double>(prob);
    }

    totalLoss /= (2.0 * batch);
    output->p(0, totalLoss);
}

void contrastiveLossBp(NDArray* imageEmbeddings, NDArray* textEmbeddings,
                          NDArray* dLdImage, NDArray* dLdText,
                          double temperature, LaunchContext* context) {
    auto batch = imageEmbeddings->sizeAt(0);
    auto embedDim = imageEmbeddings->sizeAt(1);

    // Recompute logits
    NDArray* textT = textEmbeddings->transpose();
    std::vector<sd::LongType> logitsShape2 = {batch, batch};
    NDArray logits('c', logitsShape2, imageEmbeddings->dataType());
    MmulHelper::mmul(imageEmbeddings, textT, &logits);
    delete textT;

    for (sd::LongType i = 0; i < batch; i++) {
        for (sd::LongType j = 0; j < batch; j++) {
            logits.p(i, j, logits.e<double>(i, j) * temperature);
        }
    }

    // Compute softmax along rows and columns
    NDArray rowSoftmaxMat('c', logitsShape2, imageEmbeddings->dataType());
    NDArray colSoftmaxMat('c', logitsShape2, imageEmbeddings->dataType());

    for (sd::LongType i = 0; i < batch; i++) {
        std::vector<double> rowData(batch);
        for (sd::LongType j = 0; j < batch; j++) {
            rowData[j] = logits.e<double>(i, j);
        }
        rowSoftmax(rowData.data(), batch);
        for (sd::LongType j = 0; j < batch; j++) {
            rowSoftmaxMat.p(i, j, rowData[j]);
        }
    }

    for (sd::LongType j = 0; j < batch; j++) {
        std::vector<double> colData(batch);
        for (sd::LongType i = 0; i < batch; i++) {
            colData[i] = logits.e<double>(i, j);
        }
        rowSoftmax(colData.data(), batch);
        for (sd::LongType i = 0; i < batch; i++) {
            colSoftmaxMat.p(i, j, colData[i]);
        }
    }

    // dL/dlogits combined from row and col CE
    NDArray dLdLogits('c', logitsShape2, imageEmbeddings->dataType());
    double scale = 1.0 / (2.0 * batch);

    for (sd::LongType i = 0; i < batch; i++) {
        for (sd::LongType j = 0; j < batch; j++) {
            double target = (i == j) ? 1.0 : 0.0;
            double grad = (rowSoftmaxMat.e<double>(i, j) - target +
                           colSoftmaxMat.e<double>(i, j) - target) * scale;
            dLdLogits.p(i, j, grad * temperature);
        }
    }

    // dL/dImage = dL/dLogits @ textEmbeddings
    MmulHelper::mmul(&dLdLogits, textEmbeddings, dLdImage);

    // dL/dText = dL/dLogits^T @ imageEmbeddings
    NDArray* dLdLogitsT = dLdLogits.transpose();
    MmulHelper::mmul(dLdLogitsT, imageEmbeddings, dLdText);
    delete dLdLogitsT;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
