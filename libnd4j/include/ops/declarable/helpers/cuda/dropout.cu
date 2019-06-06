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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/dropout.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void dropoutSimple(NDArray const* input, NDArray* output, double probValue, int seed) {

    }
    BUILD_SINGLE_TEMPLATE(template void dropoutSimple, (NDArray const* input, NDArray* output, double probValue, int seed), FLOAT_TYPES);

    template <typename T>
    int _dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {

        return Status::OK();
    }

    int dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, return _dropOutFunctor, (context, input, output, reduceShape, seed, probValue), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _dropOutFunctor, (graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue);, FLOAT_TYPES);

/////////////////////////////////// backrpopagations ///////////////////////////////////////////////
    template <typename T>
    static int dropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        return Status::OK();
    }

    template <typename T>
    static int alphaDropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output,
                            NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {
        return Status::OK();
    }

    template <typename T>
    int alphaDropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                              NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {

        return Status::OK();
    }

    int dropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_, (context, input, gradOut, output, reduceShape, seed, probValue), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int dropOutFunctorBP_, (graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue), FLOAT_TYPES);

    int alphaDropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_, (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int alphaDropOutFunctor_, (graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta), FLOAT_TYPES);

    int alphaDropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctorBP_, (context, input, gradOut, output, reduceShape, seed, probValue, alpha, alpha1, beta), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int alphaDropOutFunctorBP_, (graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta), FLOAT_TYPES);

}
}
}