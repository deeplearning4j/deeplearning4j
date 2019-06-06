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
// Created by raver119 on 24.01.18.
//

#ifndef LIBND4J_PAIR_H
#define LIBND4J_PAIR_H

#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT Pair {
    protected:
        int _first = 0;
        int _second = 0;

    public:
        Pair(int first = 0, int second = 0);
        ~Pair() = default;

        int first() const;
        int second() const;
    };
}


#endif //LIBND4J_PAIR_H
