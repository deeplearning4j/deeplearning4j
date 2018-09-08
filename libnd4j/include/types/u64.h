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
#ifndef LIBND4J_U64_H
#define LIBND4J_U64_H

#include <cstdint>
#include <pointercast.h>
#include <types/float16.h>


namespace nd4j {
    typedef struct {
        int16_t _v0;
        int16_t _v1;
        int16_t _v2;
        int16_t _v3;
    } di16;

    typedef struct {
        int _v0;
        int _v1;
    } di32;

    typedef struct {
        uint32_t _v0;
        uint32_t _v1;
    } du32;

    union u64 {
        bool _bool;
        int8_t _char;
        int16_t _short;
        int32_t _int;
        float16 _half = 0.0f;
        float _float;
        double _double;
        Nd4jLong _long;
        uint64_t _ulong;
    };
}

#endif