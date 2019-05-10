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

#ifndef LIBND4J_UINT16_H
#define LIBND4J_UINT16_H

#include <stdint.h>
#include <op_boilerplate.h>


namespace nd4j {

    uint16_t _CUDA_HD FORCEINLINE cpu_float2uint16(float data);
    float _CUDA_HD FORCEINLINE cpu_uint162float(uint16_t data);

    struct uint16 {
        uint16_t data;

        _CUDA_HD FORCEINLINE uint16();
        _CUDA_HD FORCEINLINE ~uint16();

        template <class T>
        _CUDA_HD FORCEINLINE uint16(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE uint16& operator=(const T& rhs);

        _CUDA_HD FORCEINLINE operator float() const;

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };

//////////////////// IMPLEMENTATIONS

    float _CUDA_HD cpu_uint162float(uint16_t data) {
        return static_cast<float>(data);
    }

    uint16_t _CUDA_HD cpu_float2uint16(float data) {
        auto t = static_cast<int>(data);
        if (t > 65536 ) t = 65536;
        if (t < 0) t = 0;

        return static_cast<uint16_t>(t);
    }

    _CUDA_HD uint16::uint16() {
        data = cpu_float2uint16(0.0f);
    }

    _CUDA_HD uint16::~uint16() {
        //
    }

    template <class T>
    _CUDA_HD uint16::uint16(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    _CUDA_HD uint16& uint16::operator=(const T& rhs) {
        assign(rhs);
        return *this;
    }

    _CUDA_HD uint16::operator float() const {
        return cpu_uint162float(data);
    }

    _CUDA_HD void uint16::assign(float rhs) {
        data = cpu_float2uint16(rhs);
    }

    _CUDA_HD void uint16::assign(double rhs) {
        assign((float)rhs);
    }
}

#endif //LIBND4J_UINT16_H
