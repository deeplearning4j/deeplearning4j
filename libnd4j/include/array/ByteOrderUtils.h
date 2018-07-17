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
// Created by raver119 on 21.11.17.
//

#ifndef LIBND4J_BYTEORDERUTILS_H
#define LIBND4J_BYTEORDERUTILS_H

#include <graph/generated/array_generated.h>
#include "ByteOrder.h"

namespace nd4j {
    class ByteOrderUtils {
    public:
        static ByteOrder fromFlatByteOrder(nd4j::graph::ByteOrder order);
    };
}


#endif //LIBND4J_BYTEORDERUTILS_H
