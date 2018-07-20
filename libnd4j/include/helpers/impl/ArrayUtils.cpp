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

#include <helpers/ArrayUtils.h>

namespace nd4j {
    namespace ArrayUtils {
        void toIntPtr(std::initializer_list<int> list, int* target) {
            std::vector<int> vec(list);
            toIntPtr(vec, target);
        }

        void toIntPtr(std::vector<int>& list, int* target) {
            memcpy(target, list.data(), list.size() * sizeof(int));
        }

        void toLongPtr(std::initializer_list<Nd4jLong> list, Nd4jLong* target) {
            std::vector<Nd4jLong> vec(list);
            toLongPtr(vec, target);
        }

        void toLongPtr(std::vector<Nd4jLong>& list, Nd4jLong* target) {
            memcpy(target, list.data(), list.size() * sizeof(Nd4jLong));
        }

        std::vector<Nd4jLong> toLongVector(std::vector<int> vec) {
            std::vector<Nd4jLong> result(vec.size());

            for (Nd4jLong e = 0; e < vec.size(); e++)
                result[e] = vec[e];

            return result;
        }

        std::vector<Nd4jLong> toLongVector(std::vector<Nd4jLong> vec) {
            return vec;
        }
    }
}
