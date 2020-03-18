/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_randomuniform)

#include <ops/declarable/CustomOperations.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/helpers/random.h>

namespace sd {
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
        CUSTOM_OP_IMPL(randomuniform, 1, 1, true, 0, 0) {
            // uniform distribution
            auto rng = block.randomGenerator();
            auto dtype = DataType::FLOAT32;
            if (block.getIArguments()->size())
                dtype = (DataType)INT_ARG(0);

            auto min = block.width() > 1 ? INPUT_VARIABLE(1) : (NDArray*) nullptr;
            auto max = block.width() > 2 ? INPUT_VARIABLE(2) : (NDArray*) nullptr;
            bool disposable = false;

            if (min == nullptr && max == nullptr && block.numT() >= 2) {
                min = NDArrayFactory::create_(dtype);
                max = NDArrayFactory::create_(dtype);
                min->p(0, T_ARG(0));
                max->p(0, T_ARG(1));
                disposable = true;
            }

            auto output = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(output->dataType() == dtype, 0, "RandomUniform: data type of output should be equals to given.");

            helpers::fillRandomUniform(block.launchContext(), rng, min, max, output);

            if (disposable) {
                delete min;
                delete max;
            }
            return Status::OK();
        }


        DECLARE_SHAPE_FN(randomuniform) {
            auto in = INPUT_VARIABLE(0);
            //auto min = INPUT_VARIABLE(1);
            auto shape = in->template asVectorT<Nd4jLong>();
            auto dtype = DataType::FLOAT32; //ArrayOptions::dataType(inputShape->at(1)); // output type is by given min

            if (block.getIArguments()->size())
                dtype = (DataType)INT_ARG(0);
            if (block.width() > 1)
                REQUIRE_TRUE(dtype == INPUT_VARIABLE(1)->dataType(), 0, "RandomUniform: data type of output and min/max args should be the same");

            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dtype, 'c', shape);
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(randomuniform) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS})
                    ->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
        }
    }
}

#endif