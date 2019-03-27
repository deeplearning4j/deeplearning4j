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
// Created by raver119 on 29/10/17.
//
// Modified by GS <sgazeos@gmail.com> 2/16/18
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lrn)

#include <ops/declarable/helpers/lrn.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

        DECLARE_TYPES(lrn) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        CONFIGURABLE_OP_IMPL(lrn, 1, 1, true, 3, 1) {
            auto input  = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn: Input rank of 4 expected, but got %i instead", input->rankOf());

            // FIXME: double?
            double alpha = T_ARG(1);
            double beta = T_ARG(2);
            double bias = T_ARG(0);
            int depth = INT_ARG(0);

            return helpers::lrnFunctor(block, input, output, depth, bias, alpha, beta);
        }

        DECLARE_TYPES(lrn_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        CONFIGURABLE_OP_IMPL(lrn_bp, 2, 1, true, 3, 1) {
            auto input  = INPUT_VARIABLE(0);
            auto errors  = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn_bp: Input rank of 4 expected, but got %i instead", input->rankOf());
            REQUIRE_TRUE(input->isSameShape(errors), 0, "lrn_bp: Both input and errors should have the same shape");

            // FIXME: double?
            double alpha = T_ARG(1);
            double beta = T_ARG(2);
            double bias = T_ARG(0);
            int depth = INT_ARG(0);

            std::unique_ptr<NDArray> unitScale(output->dup('c'));
            std::unique_ptr<NDArray> scale(output->dup('c'));

            REQUIRE_TRUE(ND4J_STATUS_OK == helpers::lrnFunctorEx(block, input, output, scale.get(), depth, bias, alpha, beta), 0, "lrn_bp: Failed to get lrn for given input." );
            //output->printBuffer("Output stage 0");
            output->applyPairwiseTransform(pairwise::Divide, input, unitScale.get(), nullptr); // 1/(b + %alpha Sum x_j ^ 2) ^ beta
            //unitScale->applyPairwiseTransform(pairwise::Multiply, scale.get(), output, nullptr);
            //output->applyPairwiseTransform(pairwise::Multiply, errors, output, nullptr);
            //output->applyPairwiseTransform(pairwise::Divide, scale.get(), output, nullptr);
            unitScale->applyPairwiseTransform(pairwise::Subtract, scale.get(), output, nullptr);
//            errors->applyPairwiseTransform(pairwise::Multiply, scale.get(), scale.get(), nullptr);
//            output->applyPairwiseTransform(pairwise::Multiply, input, output, nullptr);
            //unitScale->printBuffer("Output stage 1");
//            unitScale->applyScalar(scalar::Multiply, 2.0 * alpha * (beta), nullptr, nullptr);
           // output->printBuffer("Output stage 2");
            output->applyPairwiseTransform(pairwise::Multiply, errors, output, nullptr);
//            output->printBuffer("Output stage 3");

            return Status::OK();
        }
        DECLARE_SYN(local_response_normalization, lrn);

        CUSTOM_OP_IMPL(lrn_old, 1, 3, true, 4, 0) {
            // LocalResponseNormalization
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);
            auto unitScale = OUTPUT_VARIABLE(1);
            auto scale = OUTPUT_VARIABLE(2);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input rank of 4 expected, but got %i instead", input->rankOf());

            // FIXME: double
            double alpha = T_ARG(0);
            double beta = T_ARG(1);
            double bias = T_ARG(2);
            double depth = T_ARG(3);

            return helpers::lrnFunctorEx(block, input, output, unitScale, scale, (int)depth, bias, alpha, beta);
        }
        DECLARE_SYN(LRN, lrn_old);

        DECLARE_TYPES(lrn_old) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
        
        DECLARE_SHAPE_FN(lrn_old) {
            auto inp = inputShape->at(0);

            auto shapeList = SHAPELIST();
            for(int e = 0; e < 3; e++) {
                Nd4jLong *newShape;
                COPY_SHAPE(inp, newShape);
                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}

#endif