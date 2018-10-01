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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyvalue)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(clipbyvalue, 1, 1, true, 2, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // FIXME: extra args!!!
            auto left = T_ARG(0);
            auto right = T_ARG(1);

            REQUIRE_TRUE(left < right, 0, "clip_by_value: left bound should be lesser than right. But %f >= %f given.", left, right);
            //input->applyTransform(transform::ClipByValue, output, block.getTArguments()->data());
            helpers::clipByValue(*input, left, right, *output);
            //STORE_RESULT(*output);

            return Status::OK();
        }
        DECLARE_SYN(ClipByValue, clipbyvalue);
    }
}

#endif