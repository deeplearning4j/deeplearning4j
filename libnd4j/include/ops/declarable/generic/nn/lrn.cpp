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
        CONFIGURABLE_OP_IMPL(lrn, 1, 1, true, 3, 1) {
            auto input  = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn: Input rank of 4 expected, but got %i instead", input->rankOf());

            T alpha = T_ARG(1);
            T beta = T_ARG(2);
            T bias = T_ARG(0);
            int depth = INT_ARG(0);

            return helpers::lrnFunctor(input, output, depth, bias, alpha, beta);
        }

        CONFIGURABLE_OP_IMPL(lrn_bp, 2, 1, true, 3, 1) {
            auto input  = INPUT_VARIABLE(0);
            auto errors  = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn_bp: Input rank of 4 expected, but got %i instead", input->rankOf());
            REQUIRE_TRUE(input->isSameShape(errors), 0, "lrn_bp: Both input and errors should have the same shape");
            T alpha = T_ARG(1);
            T beta = T_ARG(2);
            T bias = T_ARG(0);
            int depth = INT_ARG(0);

            std::unique_ptr<NDArray> unitScale(errors->dup('c'));
            std::unique_ptr<NDArray> scale(errors->dup('c'));

            REQUIRE_TRUE(ND4J_STATUS_OK == helpers::lrnFunctorEx(input, output, unitScale.get(), scale.get(), (long)depth, bias, alpha, beta), 0, "lrn_bp: Failed to get lrn for given input." );

            errors->applyPairwiseTransform(pairwise::Multiply, scale.get(), scale.get(), nullptr);
            output->applyPairwiseTransform(pairwise::Multiply, input, output, nullptr);
            unitScale->applyScalar(scalar::Multiply, 2.0f * alpha * beta, unitScale.get(), nullptr);
            output->applyPairwiseTransform(pairwise::Divide, unitScale.get(), output, nullptr);
            scale->applyPairwiseTransform(pairwise::Subtract, output, output, nullptr);

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

            T alpha = T_ARG(0);
            T beta = T_ARG(1);
            T bias = T_ARG(2);
            T depth = T_ARG(3);

            return helpers::lrnFunctorEx(input, output, unitScale, scale, (int)depth, bias, alpha, beta);
        }
        DECLARE_SYN(LRN, lrn_old);
        
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