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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_gamma)

#include <ops/declarable/headers/random.h>
#include <ops/declarable/helpers/random.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(random_gamma, 2, 1, false, 0, 0) {
            // gamma distribution
            auto rng = block.randomGenerator();
            auto shape = INPUT_VARIABLE(0);
            auto alpha = INPUT_VARIABLE(1);
            NDArray* beta = nullptr;

            if (block.width() > 2) {
                beta = INPUT_VARIABLE(2);
                REQUIRE_TRUE(ShapeUtils::areShapesBroadcastable(*alpha, *beta), 0, "random_gamma: alpha and beta shapes should be broadcastable.");
            }

            auto output = OUTPUT_VARIABLE(0);
            auto seed = 0;

            if (block.getIArguments()->size()) {
                seed = INT_ARG(0);
            }

            rng.setSeed(seed);

            helpers::fillRandomGamma(block.launchContext(), rng, alpha, beta, output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(random_gamma) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();
            auto alphaShape = inputShape->at(1);
            auto additionalShape = alphaShape;
            if (inputShape->size() > 2) {
                auto rest = inputShape->at(2); additionalShape = nullptr;
                REQUIRE_TRUE(ShapeUtils::areShapesBroadcastable(alphaShape, rest), 0, "random_gamma: alpha and beta shapes should be broadcastable.");
                const Nd4jLong* additionalShapeBroadcasted = nullptr;
                ShapeUtils::evalBroadcastShapeInfo(alphaShape, rest, true, additionalShapeBroadcasted, block.workspace());
                additionalShape = additionalShapeBroadcasted;
            }
            auto lastDim = shape::sizeAt(alphaShape, 0);
            auto dtype = block.numD() > 0? D_ARG(0): ArrayOptions::dataType(alphaShape);
            for (auto i = 0; i < shape::rank(additionalShape); i++)
                shape.push_back(shape::sizeAt(additionalShape, i));
            auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', shape);
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
