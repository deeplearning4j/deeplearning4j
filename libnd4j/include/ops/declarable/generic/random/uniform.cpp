/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_randomuniform)

#include <ops/declarable/CustomOperations.h>
#include <helpers/RandomLauncher.h>

namespace nd4j {
    namespace ops {
        ///////////////////////
        /**
         * uniform distribution
         * takes 1 ndarray
         *
         * T argumens map:
         * TArgs[0] - min for rng
         * TArgs[1] - max for rng
         */
        CUSTOM_OP_IMPL(randomuniform, 1, 1, true, 2, 0) {
            // uniform distribution
            auto rng = block.randomGenerator();

            // FIXME: to be implemented
            /*
            if (rng == nullptr)
                return Status::THROW("RNG is null, aborting...");

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());

            STORE_RESULT(*z);
*/
            REQUIRE_TRUE(block.numT() > 1, 0, "RandomUniform: to/from must be set");

            RandomLauncher::fillUniform(rng, OUTPUT_VARIABLE(0), T_ARG(0), T_ARG(1));
            return Status::OK();
        }


        DECLARE_SHAPE_FN(randomuniform) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();

            Nd4jLong *newShape = nd4j::ShapeBuilders::createShapeInfo(block.dataType(), 'c', shape, block.getWorkspace());            

            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(randomuniform) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif