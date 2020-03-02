/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
//

#ifndef LIBND4J_ACTIVATIONS_H
#define LIBND4J_ACTIVATIONS_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

    ND4J_EXPORT void softMaxForVector(sd::LaunchContext * context, const NDArray &input, NDArray &output);

    ND4J_EXPORT void logSoftMaxForVector(sd::LaunchContext * context, const NDArray &input, NDArray &output);

    ND4J_EXPORT void softmax(sd::LaunchContext * context, const NDArray &input, NDArray &output, const int dimension);

    ND4J_EXPORT void logSoftmax(sd::LaunchContext * context, const NDArray &input, NDArray &output, const int dimension);

    ND4J_EXPORT void softmaxDerivative(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension);

    ND4J_EXPORT void prelu(sd::LaunchContext * context, const NDArray &input, const NDArray &alpha, NDArray &output);

    ND4J_EXPORT void preluBP(sd::LaunchContext * context, const NDArray &input, const NDArray &alpha, const NDArray &dLdO, NDArray &dLdI, NDArray &dLdA);

    ND4J_EXPORT void thresholdRelu(sd::LaunchContext * context, const NDArray &input, double threshold, NDArray &output);

    ND4J_EXPORT void thresholdReluDerivative(sd::LaunchContext * context, NDArray *input, double threshold, NDArray* dLdO, NDArray *output);
}
}
}


#endif //LIBND4J_ACTIVATIONS_H
