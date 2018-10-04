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

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void _softMaxForVector(void *input, Nd4jLong *xShapeInfo, void *output, Nd4jLong *zShapeInfo);
    void softMaxForVector(const NDArray& input, NDArray& output);

    template <typename T>
    void _logSoftMaxForVector(void *input, Nd4jLong *xShapeInfo, void *output, Nd4jLong *zShapeInfo);
    void logSoftMaxForVector(const NDArray& input, NDArray& output);

    void softmax(const NDArray& input, NDArray& output, const int dimension);

    void prelu(const NDArray& input, const NDArray& alpha, NDArray& output);

    void preluBP(const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA);

    bool checkAlphaShapeLen(std::vector<Nd4jLong> const& expectedShape, Nd4jLong shapeLen);
    void thresholdRelu(const NDArray& input, double threshold, NDArray& output);
}
}
}


#endif //LIBND4J_ACTIVATIONS_H
