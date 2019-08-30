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
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_BROADCASTINTOPSTUPLE_H
#define DEV_TESTS_BROADCASTINTOPSTUPLE_H

#include <op_enums.h>

namespace nd4j {
    class BroadcastIntOpsTuple {
    private:

    public:
        nd4j::scalar::IntOps s;
        nd4j::pairwise::IntOps p;
        nd4j::broadcast::IntOps b;

        BroadcastIntOpsTuple() = default;
        ~BroadcastIntOpsTuple() = default;

        BroadcastIntOpsTuple(nd4j::scalar::IntOps scalar, nd4j::pairwise::IntOps pairwise, nd4j::broadcast::IntOps broadcast) {
            s = scalar;
            p = pairwise;
            b = broadcast;
        }

        static BroadcastIntOpsTuple custom(nd4j::scalar::IntOps scalar, nd4j::pairwise::IntOps pairwise, nd4j::broadcast::IntOps broadcast);
    };
}


#endif //DEV_TESTS_BROADCASTOPSTUPLE_H
