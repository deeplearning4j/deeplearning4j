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
//  @author raver119@protonmail.com
//

#ifndef LIBND4J_GRAPH_RNG_H
#define LIBND4J_GRAPH_RNG_H

#include <types/u64.h>
#include <pointercast.h>
#include <op_boilerplate.h>

namespace nd4j {
    namespace graph {
        class RandomGenerator {
        private:
            // GRAPH-LEVEL STATE
            u64 _rootState;

            // NODE-LEVEL STATE
            u64 _nodeState;

            /**
             * Utility method, returns number of milliseconds since 1970
             */
            Nd4jLong currentMilliseconds();


            uint32_t xoroshiro32(Nd4jLong index);
            uint64_t xoroshiro64(Nd4jLong index);

            /**
             * This method returns integer value between 0 and MAX_UINT
             */
            uint32_t relativeInt(Nd4jLong index);

        public:
            RandomGenerator(Nd4jLong rootSeed = 0, Nd4jLong nodeSeed = 0);
            ~RandomGenerator();

            /**
             * This method allows to change graph-level state in runtime.
             * PLEASE NOTE: this method will change state of node as well.
             */
            void setStates(Nd4jLong rootSeed, Nd4jLong nodeState = 0);

            

            /**
             * This method returns T value between from and to
             */
            template <typename T>
            T relativeT(Nd4jLong index, T from, T to);

            /**
             * This method returns T value between 0 and MAX_T
             */
            template <typename T>
            T relativeT(Nd4jLong index);


            void rewindH(Nd4jLong steps);
            void setSeed(uint64_t seed) { _nodeState._ulong = seed; }
        };
    }
}

#endif
