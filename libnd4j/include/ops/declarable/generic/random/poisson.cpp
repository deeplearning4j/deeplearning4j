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
#if NOT_EXCLUDED(OP_random_poisson)

#include <ops/declarable/headers/random.h>
#include <helpers/RandomLauncher.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(random_poisson, 2, 1, false, 0, 0) {
            // gamma distribution
            auto rng = block.randomGenerator();
            auto shape = INPUT_VARIABLE(0);
            auto lambda = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);
            auto seed = 0;
            if (block.getIArguments()->size()) {
                seed = INT_ARG(0);
            }

            //RandomLauncher::fillExponential(block.launchContext(), rng, z, lambda);

            return Status::OK();
        }


        DECLARE_SHAPE_FN(random_poisson) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();
            auto lambdaShape = inputShape->at(1);
            auto lastDim = shape::sizeAt(lambdaShape, 0);
            auto dtype = ArrayOptions::dataType(lambdaShape);
            shape.push_back(lastDim);
            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dtype, 'c', shape);
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(random_poisson) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif