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
// Stronger 64-bit hash function helper, as described here: http://www.javamex.com/tutorials/collections/strong_hash_code_implementation.shtml
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_HASH_H
#define LIBND4J_HELPER_HASH_H

#include <string>
#include <dll.h>
#include <pointercast.h>
#include <mutex>

namespace nd4j {
    namespace ops {
        class ND4J_EXPORT HashHelper {
        private:
            static HashHelper* _INSTANCE;

            Nd4jLong _byteTable[256];
            const Nd4jLong HSTART = 0xBB40E64DA205B064L;
            const Nd4jLong HMULT = 7664345821815920749L;

            bool _isInit = false;
            std::mutex _locker;

        public:
            static HashHelper* getInstance();
            Nd4jLong getLongHash(std::string& str);
        };
    }
}

#endif //LIBND4J_HELPER_HASH_H
