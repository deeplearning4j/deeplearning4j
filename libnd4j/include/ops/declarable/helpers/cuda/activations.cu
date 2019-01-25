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
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/activations.h>
#include <ShapeUtils.h>
#include <numeric>

namespace nd4j    {
namespace ops     {
namespace helpers {

    template <typename T>
    void _softMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {

    }

    template <typename T>
    void _logSoftMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {

    }

    ///////////////////////////////////////////////////////////////////
    void softMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _softMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }


    ///////////////////////////////////////////////////////////////////
    void logSoftMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _logSoftMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void softmax(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {
        
            if(rank == 1 || input.sizeAt(dimension) != 1)
                softMaxForVector(context, input, output);
            else
                output = 1.;
        }
        else {
            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            auto exponents = (input - maxAlongDim).transform(transform::Exp);
            auto sumAlongDim = exponents.reduceAlongDims(reduce::Sum, {dimension}, true);

            // FIXME: assign?
            output.assign(exponents / sumAlongDim);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void prelu(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, NDArray& output) {

    }

    //////////////////////////////////////////////////////////////////////////
    void preluBP(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

    }

    BUILD_SINGLE_TEMPLATE(template void _softMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void _logSoftMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);

    bool checkAlphaShapeLen(graph::LaunchContext* context, std::vector<Nd4jLong> const& expectedShape, Nd4jLong shapeLen) {
        Nd4jLong expectedAlphaLen = std::accumulate(expectedShape.cbegin(), expectedShape.cend(), 1, std::multiplies<Nd4jLong>());
        return expectedAlphaLen == shapeLen;
    }

    template <typename T>
    static void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
        auto routine = LAMBDA_T(_x, threshold) {
            return _x > (T)threshold? _x: (T)0.f;
        };
        const_cast<NDArray&>(input).applyLambda<T>(routine, &output);
    }

    void thresholdRelu(graph::LaunchContext* context, NDArray const& input, double threshold, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
    }

    template <typename T>
    static void thresholdReluDerivative_(NDArray* input, double theta, NDArray* dLdO, NDArray* output) {

    }

    void thresholdReluDerivative(graph::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdO, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_, (NDArray* input, double threshold, NDArray* dLdO, NDArray* output), FLOAT_TYPES);

}
}
}

