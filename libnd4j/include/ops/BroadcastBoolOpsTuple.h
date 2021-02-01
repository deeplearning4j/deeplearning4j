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
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_BROADCASTBOOLOPSTUPLE_H
#define DEV_TESTS_BROADCASTBOOLOPSTUPLE_H

#include <system/op_enums.h>
#include <system/dll.h>

namespace sd {
    class ND4J_EXPORT BroadcastBoolOpsTuple {
    private:

    public:
        sd::scalar::BoolOps s;
        sd::pairwise::BoolOps p;
        sd::broadcast::BoolOps b;

        BroadcastBoolOpsTuple() = default;
        ~BroadcastBoolOpsTuple() = default;

        BroadcastBoolOpsTuple(sd::scalar::BoolOps scalar, sd::pairwise::BoolOps pairwise, sd::broadcast::BoolOps broadcast) {
            s = scalar;
            p = pairwise;
            b = broadcast;
        }

        static BroadcastBoolOpsTuple custom(sd::scalar::BoolOps scalar, sd::pairwise::BoolOps pairwise, sd::broadcast::BoolOps broadcast);
    };
}


#endif //DEV_TESTS_BROADCASTOPSTUPLE_H
