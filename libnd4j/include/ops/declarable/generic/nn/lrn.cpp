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
// @author raver119 on 29/10/17
// @author GS <sgazeos@gmail.com> 2/16/18
// @author Yurii Shyrma (iuriish@yahoo.com) -> back prop author
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
            auto input = INPUT_VARIABLE(0);
            auto gradO = INPUT_VARIABLE(1);
            auto gradI = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn_bp: Input rank of 4 expected, but got %i instead", input->rankOf());
            REQUIRE_TRUE(input->isSameShape(gradO), 0, "lrn_bp: Both input and grad_output should have the same shape, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

            // FIXME: double/float?
            float bias  = T_ARG(0);
            float alpha = T_ARG(1);
            float beta  = T_ARG(2);            
            int depth   = INT_ARG(0);

            helpers::lrnBP(block, *input, *gradO, *gradI, depth, bias, alpha, beta);

            return Status::OK();
        }
        DECLARE_SYN(local_response_normalization, lrn);
      
    }
}

#endif