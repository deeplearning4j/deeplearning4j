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
//  @author George A. Shulinok <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_gamma)

#include <ops/declarable/headers/random.h>
#include <ops/declarable/helpers/random.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(random_gamma, 2, 1, false, 0, 0) {
            // gamma distribution
            auto rng = block.randomGenerator();
            auto shape = INPUT_VARIABLE(0);
            auto alpha = INPUT_VARIABLE(1);
            NDArray* beta = nullptr;
            if (block.width() > 2) {
                beta = INPUT_VARIABLE(2);
                REQUIRE_TRUE(alpha->isSameShape(beta), 0, "random_gamma: alpha and beta shapes should be equals.");
            }
            auto output = OUTPUT_VARIABLE(0);
            auto seed = 0;
            if (block.getIArguments()->size()) {
                seed = INT_ARG(0);
            }
            rng.setSeed(seed);
            helpers::fillRandomGamma(block.launchContext(), rng, alpha, beta, output);
            //RandomLauncher::fillExponential(block.launchContext(), rng, z, lambda);

            return Status::OK();
        }


        DECLARE_SHAPE_FN(random_gamma) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();
            auto alphaShape = inputShape->at(1);
            auto lastDim = shape::sizeAt(alphaShape, 0);
            auto dtype = ArrayOptions::dataType(alphaShape);
            for (auto i = 0; i < shape::rank(alphaShape); i++)
                shape.push_back(shape::sizeAt(alphaShape, i));
            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dtype, 'c', shape);
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(random_gamma) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif