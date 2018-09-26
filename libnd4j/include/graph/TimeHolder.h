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
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_TIMEHOLDER_H
#define LIBND4J_TIMEHOLDER_H

#include <map>
#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT TimeHolder {
        private:
            std::map<int, Nd4jLong> _outer;
            std::map<int, Nd4jLong> _inner;


        public:

            TimeHolder() = default;
            ~TimeHolder() = default;


            void setOuterTime(int nodeId, Nd4jLong time);
            void setInnerTime(int nodeId, Nd4jLong time);


            Nd4jLong outerTime(int nodeId);
            Nd4jLong innerTime(int nodeId);
        };
    }
}

#endif //LIBND4J_TIMEHOLDER_H
