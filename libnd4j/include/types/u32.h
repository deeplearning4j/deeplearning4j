/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
#ifndef SD_U32_H
#define SD_U32_H

#include <cstdint>
#include <system/pointercast.h>


namespace sd {
    union u32 {
        bool _bool;
        int8_t _s8;
        uint8_t _u8;
        int16_t _s16;
        uint16_t _u16;
        int32_t _s32;
        uint32_t _u32;
        float _f32;
    };
}

#endif