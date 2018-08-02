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
// Created by raver on 6/12/2018.
//

#ifndef LIBND4J_TYPES_H
#define LIBND4J_TYPES_H

#include <pointercast.h>

//// Forward declarations of custom types
struct float16;
namespace nd4j {
    struct float8;
    struct int8;
    struct uint8;
    struct int16;
    struct uint16;
}

#define LIBND4J_TYPES \
        float, \
        float16, \
        nd4j::float8, \
        double, \
        int, \
        Nd4jLong, \
        nd4j::int8, \
        nd4j::uint8, \
        nd4j::int16, \
        nd4j::uint16

#endif //LIBND4J_TYPES_H
