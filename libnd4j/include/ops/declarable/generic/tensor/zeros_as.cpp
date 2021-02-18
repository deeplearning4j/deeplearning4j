/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 12.10.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_zeros_as)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(zeros_as, 1, 1, false, 0, 0) {
            auto out = OUTPUT_VARIABLE(0);

            out->assign(0); // output is filled by zero by default

            return Status::OK();
        }
        DECLARE_SYN(zeroslike, zeros_as);
        DECLARE_SYN(zeros_like, zeros_as);


        DECLARE_SHAPE_FN(zeros_as) {
            auto in = inputShape->at(0);
            auto dtype = block.numD() ? D_ARG(0) : ArrayOptions::dataType(in);
            auto shape = sd::ConstantShapeHelper::getInstance().createShapeInfo(dtype, in);

            return SHAPELIST(shape);
        }

        DECLARE_TYPES(zeros_as) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes(sd::DataType::ANY)
                    ->setSameMode(false);
        }
    }
}

#endif