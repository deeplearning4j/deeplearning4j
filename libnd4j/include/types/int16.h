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

#ifndef LIBND4J_INT16_H
#define LIBND4J_INT16_H

#include <stdint.h>
#include <op_boilerplate.h>
#include <types/types.h>


namespace nd4j {

    float _CUDA_HD FORCEINLINE cpu_int162float(int16_t data);
    double _CUDA_HD FORCEINLINE cpu_int162double(int16_t data);
    int16_t _CUDA_HD FORCEINLINE cpu_float2int16(float data);

    struct int16 {
        int16_t data;

        _CUDA_HD FORCEINLINE int16();
        _CUDA_HD FORCEINLINE ~int16() = default;

        template <class T>
        _CUDA_HD FORCEINLINE int16(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE int16& operator=(const T& rhs);


        //// INTEGER CASTING ////

        _CUDA_HD operator int8() const;

        _CUDA_HD operator uint8() const;

        _CUDA_HD operator uint16() const;

        _CUDA_HD FORCEINLINE explicit operator int() const;

        _CUDA_HD FORCEINLINE explicit operator Nd4jLong() const;

        //// INTEGER ASSIGNING ////

        _CUDA_HD FORCEINLINE void assign(int16_t rhs);

        _CUDA_HD void assign(const int8&  rhs);

        _CUDA_HD void assign(const uint8& rhs);

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


    //////////////////////////////

    float cpu_int162float(int16_t data) {
        return static_cast<float>(static_cast<int>(data));
    }
    double cpu_int162double(int16_t data) {
        return static_cast<double>(static_cast<int>(data));
    }

    int16_t cpu_float2int16(float data) {
        auto t = static_cast<int>(data);
        if (t > 32767 ) t = 32767;
        if (t < -32768) t = -32768;

        return static_cast<int16_t>(t);
    }


    int16::int16() {
        data = cpu_float2int16(0.0f);
    }

    template <class T>
    int16::int16(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    int16& int16::operator=(const T& rhs) {
        assign(rhs); return *this;
    }

    ///////  CAST INT TYPES

    int16::operator int() const {
        return static_cast<int>(data);
    }

    int16::operator Nd4jLong() const {
        return static_cast<Nd4jLong>(data);
    }

    ///////  ASSIGN INT TYPES
    void int16::assign(int16_t rhs) {
        data = rhs;
    }

    void int16::assign(int rhs) {
        assign(static_cast<int16_t>(rhs));
    }

    void int16::assign(Nd4jLong rhs) {
        assign(static_cast<int16_t>(rhs));
    }

    ///////  CAST FLOAT TYPES

    int16::operator float() const {
        return cpu_int162float(data);
    }

    int16::operator double() const {
        return cpu_int162double(data);
    }

    ///////  ASSIGN FLOAT TYPES

    void int16::assign(double rhs) {
        assign(static_cast<float>(rhs));
    }


    void int16::assign(float rhs) {
        data = cpu_float2int16(rhs);
    }

}

#endif //LIBND4J_INT16_H
