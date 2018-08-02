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

#ifndef LIBND4J_INT8_H
#define LIBND4J_INT8_H

#include <stdint.h>
#include <op_boilerplate.h>
#include <types/types.h>


namespace nd4j {

    float _CUDA_HD FORCEINLINE cpu_int82float(int8_t data);
    double _CUDA_HD FORCEINLINE cpu_int82double(int8_t data);
    int8_t _CUDA_HD FORCEINLINE cpu_float2int8(float data);

    struct int8 {
        int8_t data;

        _CUDA_HD FORCEINLINE int8();
        _CUDA_HD FORCEINLINE ~int8() = default;

        template <class T>
        _CUDA_HD FORCEINLINE int8(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE int8& operator=(const T& rhs);

        //// INTEGER CASTING ////

        _CUDA_HD operator uint8() const;

        _CUDA_HD operator int16() const;

        _CUDA_HD operator uint16() const;

        _CUDA_HD FORCEINLINE explicit operator int() const;

        _CUDA_HD FORCEINLINE explicit operator Nd4jLong() const;

        //// INTEGER ASSIGNING ////

        _CUDA_HD FORCEINLINE void assign(int8_t rhs);

        _CUDA_HD void assign(const uint8& rhs);

        _CUDA_HD void assign(const int16&  rhs);

        _CUDA_HD void assign(const uint16&  rhs);

        _CUDA_HD FORCEINLINE void assign(int rhs);

        _CUDA_HD FORCEINLINE void assign(Nd4jLong rhs);

        //// FLOAT CASTING ////

        _CUDA_HD operator float16() const;

        _CUDA_HD operator float8() const;

        _CUDA_HD FORCEINLINE explicit operator float() const;

        _CUDA_HD FORCEINLINE explicit operator double() const;

        //// FLOAT ASSIGNING ////

        _CUDA_HD void assign(const float8& rhs);

        _CUDA_HD void assign(const float16& rhs);

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };


    float cpu_int82float(int8_t data) {
        return static_cast<float>(static_cast<int>(data));
    }
    double cpu_int82double(int8_t data) {
        return static_cast<double>(static_cast<int>(data));
    }


    int8_t cpu_float2int8(float data) {
        int t = (int) data;
        if (t > 127) t = 127;
        if (t < -128) t = -128;

        return (int8_t) t;
    }

    int8::int8() {
        data = cpu_float2int8(0.0f);
    }

    template <class T>
    int8::int8(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    int8& int8::operator=(const T& rhs) {
        assign(rhs); return *this;
    }

    ///////  CAST INT TYPES

    int8::operator int() const {
        return static_cast<int>(data);
    }

    int8::operator Nd4jLong() const {
        return static_cast<Nd4jLong>(data);
    }
    
    ///////  ASSIGN INT TYPES

    void int8::assign(int8_t rhs) {
        data = rhs;
    }

    void int8::assign(int rhs) {
        assign(static_cast<int8_t>(rhs));
    }

    void int8::assign(Nd4jLong rhs) {
        assign(static_cast<int8_t>(rhs));
    }

    ///////  CAST FLOAT TYPES

    int8::operator float() const {
        return cpu_int82float(data);
    }

    int8::operator double() const {
        return cpu_int82double(data);
    }

    ///////  ASSIGN FLOAT TYPES

    void int8::assign(double rhs) {
        assign(static_cast<float>(rhs));
    }


    void int8::assign(float rhs) {
        data = cpu_float2int8(rhs);
    }
}

#endif //LIBND4J_INT8_H
