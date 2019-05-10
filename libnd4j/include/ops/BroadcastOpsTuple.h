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

#ifndef DEV_TESTS_BROADCASTOPSTUPLE_H
#define DEV_TESTS_BROADCASTOPSTUPLE_H

#include <op_enums.h>

namespace nd4j {
    class BroadcastOpsTuple {
    private:

    public:
        nd4j::scalar::Ops  s;
        nd4j::pairwise::Ops p;
        nd4j::broadcast::Ops b;

        BroadcastOpsTuple() = default;
        ~BroadcastOpsTuple() = default;

        BroadcastOpsTuple(nd4j::scalar::Ops scalar, nd4j::pairwise::Ops pairwise, nd4j::broadcast::Ops broadcast) {
            s = scalar;
            p = pairwise;
            b = broadcast;
        }

        static BroadcastOpsTuple custom(nd4j::scalar::Ops scalar, nd4j::pairwise::Ops pairwise, nd4j::broadcast::Ops broadcast);

        static BroadcastOpsTuple Add();
        static BroadcastOpsTuple Assign();
        static BroadcastOpsTuple Divide();
        static BroadcastOpsTuple Multiply();
        static BroadcastOpsTuple Subtract();
    };
}


#endif //DEV_TESTS_BROADCASTOPSTUPLE_H
