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
// Created by raver119 on 22.11.2017.
//

#ifndef LIBND4J_FLATUTILS_H
#define LIBND4J_FLATUTILS_H

#include <utility>
#include <pointercast.h>
#include <graph/generated/array_generated.h>
#include <graph/generated/node_generated.h>
#include <NDArray.h>

namespace nd4j {
    namespace graph {
        class FlatUtils {
        public:
            static std::pair<int, int> fromIntPair(IntPair* pair);

            static std::pair<Nd4jLong, Nd4jLong> fromLongPair(LongPair* pair);

            static NDArray* fromFlatArray(const nd4j::graph::FlatArray* flatArray);
        };
    }
}

#endif //LIBND4J_FLATUTILS_H
