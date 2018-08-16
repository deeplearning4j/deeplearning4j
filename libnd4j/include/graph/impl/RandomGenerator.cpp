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
// @author raver119@protonmail.com
//
// relies on xoroshiro 64** and xoroshiro128 implementations

#include <op_boilerplate.h>
#include <pointercast.h>
#include <graph/RandomGenerator.h>
#include <chrono>
#include <array/DataTypeUtils.h>
#include <helpers/logger.h>

namespace nd4j {
    namespace graph {
        RandomGenerator::RandomGenerator(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            this->setStates(rootSeed, nodeSeed);
        }
        
        RandomGenerator::~RandomGenerator() {
            //
            // :)
        }

        void RandomGenerator::setStates(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            // this seed is used graph-level state
            if (rootSeed == 0)
                rootSeed = currentMilliseconds();

            // graph-level state is just first seed
            _rootState._long = rootSeed;

            // used to build second, node state
            _nodeState._long = nodeSeed;
        }


        Nd4jLong RandomGenerator::currentMilliseconds() {
            auto s = std::chrono::system_clock::now().time_since_epoch();
            auto v = std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
            return v;
        }


        uint64_t RandomGenerator::relativeUint64(Nd4jLong index) {
            return this->xoroshiro64(index);
        }

        uint32_t RandomGenerator::relativeUint32(Nd4jLong index) {
            return this->xoroshiro32(index);
        }



        //////
        static FORCEINLINE uint32_t rotl(const uint32_t x, int k) {
	        return (x << k) | (x >> (32 - k));
        }

        static FORCEINLINE uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        uint32_t RandomGenerator::xoroshiro32(Nd4jLong index) {
            u64 v;
            // TODO: improve this
            v._long = _rootState._long ^ _nodeState._long ^ index;

            return rotl(v._du32._v0 * 0x9E3779BB, 5) * 5;
        }

        uint64_t RandomGenerator::xoroshiro64(Nd4jLong index) {
            auto s0 = _rootState._ulong;
            auto s1 = _nodeState._ulong;

            // xor by idx
            _nodeState._long ^= index;

            // since we're not modifying state - do rotl step right here
            s1 ^= s0;
            s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
            s1 = rotl(s1, 36);

            return s0 + s1;
        }

        void RandomGenerator::rewindH(Nd4jLong steps) {
            auto s0 = _nodeState._du32._v0;
            auto s1 = _nodeState._du32._v1;

            s1 ^= s0;
	        _nodeState._du32._v0 = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
	        _nodeState._du32._v1 = rotl(s1, 13); // c

            // TODO: improve this
            _nodeState._long ^= steps;
        }
    }
}
