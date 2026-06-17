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

#include <ops/declarable/helpers/center_and_sharpen.h>
#include <helpers/MmulHelper.h>
#include <math/templatemath.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

void centerAndSharpen(NDArray* input, NDArray* center, NDArray* output,
                         double temperature, LaunchContext* context) {
    // Forward: output = softmax((input - center) / temperature) per row
    // input: [batch, dim], center: [1, dim] or [dim], output: [batch, dim]

    auto batchSize = input->sizeAt(0);
    auto dim = input->sizeAt(1);

    for (sd::LongType b = 0; b < batchSize; b++) {
        // Compute scaled centered values and find row max for numerical stability
        double rowMax = -1e30;
        for (sd::LongType d = 0; d < dim; d++) {
            double val = (input->e<double>(b, d) - center->e<double>(d)) / temperature;
            output->p(b, d, val);
            if (val > rowMax) rowMax = val;
        }

        // Compute exp(x - max) and sum
        double expSum = 0.0;
        for (sd::LongType d = 0; d < dim; d++) {
            double val = sd::math::sd_exp<double, double>(output->e<double>(b, d) - rowMax);
            output->p(b, d, val);
            expSum += val;
        }

        // Normalize
        for (sd::LongType d = 0; d < dim; d++) {
            output->p(b, d, output->e<double>(b, d) / expSum);
        }
    }
}

void centerAndSharpenBp(NDArray* input, NDArray* center, NDArray* gradOutput,
                           NDArray* dLdInput, NDArray* dLdCenter,
                           double temperature, LaunchContext* context) {
    // Backward through softmax((input - center) / temperature)
    // Let s = softmax(z) where z = (input - center) / temperature
    // dsoftmax/dz = diag(s) - s * s^T  (Jacobian)
    // dL/dz = s * (gradOutput - sum(gradOutput * s))  (per row)
    // dL/d(input) = dL/dz / temperature
    // dL/d(center) = -dL/dz / temperature  (summed over batch)

    auto batchSize = input->sizeAt(0);
    auto dim = input->sizeAt(1);

    // First compute the forward softmax output (needed for backward)
    NDArray softmaxOut(input->shapeInfo(), false, const_cast<LaunchContext*>(context), true);
    centerAndSharpen(input, center, &softmaxOut, temperature, context);

    dLdCenter->nullify();

    for (sd::LongType b = 0; b < batchSize; b++) {
        // Compute dot(gradOutput[b], softmax[b])
        double dotProduct = 0.0;
        for (sd::LongType d = 0; d < dim; d++) {
            dotProduct += gradOutput->e<double>(b, d) * softmaxOut.e<double>(b, d);
        }

        // dL/dz[b] = softmax[b] * (gradOutput[b] - dot)
        for (sd::LongType d = 0; d < dim; d++) {
            double s = softmaxOut.e<double>(b, d);
            double grad = gradOutput->e<double>(b, d);
            double dLdz = s * (grad - dotProduct) / temperature;

            dLdInput->p(b, d, dLdz);

            // Accumulate negative gradient for center
            double prevCenter = dLdCenter->e<double>(d);
            dLdCenter->p(d, prevCenter - dLdz);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
