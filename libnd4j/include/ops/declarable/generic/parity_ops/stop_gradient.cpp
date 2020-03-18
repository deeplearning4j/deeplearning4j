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
#if NOT_EXCLUDED(OP_stop_gradient)

#include <ops/declarable/headers/parity_ops.h>

namespace sd {
    namespace ops {
        OP_IMPL(stop_gradient, 1, 1, true) {
            auto out = OUTPUT_VARIABLE(0);

            if (!block.isInplace()) {
                auto x = INPUT_VARIABLE(0);
                // we hope for memcpy here
                out->assign(x);
            }

            return Status::OK();
        }
        DECLARE_SYN(StopGradient, stop_gradient);

        DECLARE_TYPES(stop_gradient) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setSameMode(true);
        }
    }
}

#endif