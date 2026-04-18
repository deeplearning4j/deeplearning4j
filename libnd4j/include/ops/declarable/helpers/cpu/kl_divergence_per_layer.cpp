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

//
// CPU implementation of per-layer KL divergence for adaptive quantization.
//
// For each row r of the (batch x dim) logit matrices:
//   P_r = softmax(ref_r  / temperature)
//   Q_r = softmax(quant_r / temperature)
//   kl_r = sum_c P_r[c] * log(P_r[c] / Q_r[c])
// output = mean(kl_r)
//

#include <ops/declarable/helpers/kl_divergence_per_layer.h>
#include <system/openmp_pragmas.h>
#include <cmath>
#include <algorithm>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Numerically-stable in-place log-softmax for a single row.
// Divides by temperature before computing softmax.
static void logSoftmaxRow(const double* src, double* dst, LongType dim, double temperature) {
    double maxVal = src[0] / temperature;
    for (LongType c = 1; c < dim; c++) {
        double v = src[c] / temperature;
        if (v > maxVal) maxVal = v;
    }
    double logSum = 0.0;
    for (LongType c = 0; c < dim; c++) {
        dst[c] = src[c] / temperature - maxVal;
        logSum += std::exp(dst[c]);
    }
    logSum = std::log(logSum);
    for (LongType c = 0; c < dim; c++) {
        dst[c] -= logSum;
    }
}

void klDivergencePerLayer(NDArray* referenceLogits,
                           NDArray* quantizedLogits,
                           NDArray* output,
                           double temperature,
                           LaunchContext* context) {

    // Flatten to 2D: [numRows, dim] — collapse all leading dimensions
    LongType dim = referenceLogits->sizeAt(referenceLogits->rankOf() - 1);
    LongType numRows = referenceLogits->lengthOf() / dim;

    if (temperature <= 0.0) temperature = 1.0;

    double totalKL = 0.0;

    std::vector<double> refRow(dim);
    std::vector<double> quantRow(dim);
    std::vector<double> logP(dim);
    std::vector<double> logQ(dim);

    for (LongType r = 0; r < numRows; r++) {
        LongType offset = r * dim;

        // Read row from NDArray (handles arbitrary strides)
        for (LongType c = 0; c < dim; c++) {
            refRow[c]   = referenceLogits->e<double>(offset + c);
            quantRow[c] = quantizedLogits->e<double>(offset + c);
        }

        logSoftmaxRow(refRow.data(),   logP.data(), dim, temperature);
        logSoftmaxRow(quantRow.data(), logQ.data(), dim, temperature);

        // KL(P||Q) = sum_c P_c * (log P_c - log Q_c)
        double kl = 0.0;
        for (LongType c = 0; c < dim; c++) {
            double p = std::exp(logP[c]);
            if (p > 1e-12) {
                kl += p * (logP[c] - logQ[c]);
            }
        }
        totalKL += kl;
    }

    double meanKL = numRows > 0 ? totalKL / static_cast<double>(numRows) : 0.0;
    output->p(0, meanKL);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
