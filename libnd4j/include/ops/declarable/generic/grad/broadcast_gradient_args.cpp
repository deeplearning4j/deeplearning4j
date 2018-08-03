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
#if NOT_EXCLUDED(OP_broadcastgradientargs)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
         OP_IMPL(broadcastgradientargs, 2, 2, true) {
            
            nd4j_printf("BroadcastGradientArgs: Not implemented yet\n", "");

            return ND4J_STATUS_KERNEL_FAILURE;
        }
        DECLARE_SYN(BroadcastGradientArgs, broadcastgradientargs);
    }
}

#endif