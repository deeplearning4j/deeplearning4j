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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lrelu)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>
namespace sd {
    namespace ops {
        CONFIGURABLE_OP_IMPL(lrelu, 1, 1, true, -2, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            float alpha = block.numT() > 0 ? T_ARG(0) : 0.01f;

            input->applyScalar(sd::scalar::LeakyRELU, alpha, *output);
            STORE_RESULT(output);

            return Status::OK();
        }

        DECLARE_TYPES(lrelu) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedOutputTypes(0, {ALL_FLOATS});
        }

        CONFIGURABLE_OP_IMPL(lrelu_bp, 2, 1, true, -2, 0) {
            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            float alpha = block.numT() > 0 ? T_ARG(0) : 0.01f;

            //input->applyPairwiseTransform(pairwise::LRELUDerivativeE, epsilon, z, nullptr);
            helpers::leakyReluDerivative(block.launchContext(), input, epsilon, z, alpha);
            return Status::OK();
        }

        DECLARE_TYPES(lrelu_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedInputTypes(1, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF})
                    ->setAllowedOutputTypes(0, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF});
        }
    }
}

#endif