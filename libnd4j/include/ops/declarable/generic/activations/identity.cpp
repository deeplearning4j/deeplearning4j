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
#if NOT_EXCLUDED(OP_identity)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(identity, 1, 1, true) {
            auto first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            // just for lulz
            first->applyTransform(nd4j::transform::Identity, z, nullptr);

            STORE_RESULT(*z);

            return Status::OK();
        }
        DECLARE_SYN(linear, identity);


        OP_IMPL(identity_bp, 2, 1, true) {
            auto first = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            z->assign(epsilon);

            return Status::OK();
        }
        DECLARE_SYN(LinearGrad, identity_bp);
    }
}

#endif