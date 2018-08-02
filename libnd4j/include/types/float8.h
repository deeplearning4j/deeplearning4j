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
// Created by raver119 on 10.08.16.
//
#include <limits>
#ifndef LIBND4J_FLOAT8_H
#define LIBND4J_FLOAT8_H

/*
#ifdef __CUDACC__
#define local_def __host__ __device__
#elif _MSC_VER
#define local_def
#elif __clang__
#define local_def
#elif __GNUC__
#define local_def
#endif
*/

#include <op_boilerplate.h>
#include <types/types.h>


namespace nd4j {

    typedef struct {
        unsigned char x;
    } __quarter;

    typedef __quarter quarter;

    quarter _CUDA_HD FORCEINLINE cpu_float2quarter_rn(float f);
    float _CUDA_HD FORCEINLINE  cpu_quarter2float(quarter b);

    int _CUDA_HD FORCEINLINE  cpu_quarter2int(quarter b);

    struct float8 {
        quarter data;

        _CUDA_HD FORCEINLINE float8();
        _CUDA_HD FORCEINLINE float8(quarter data);

        template <class T>
        _CUDA_HD FORCEINLINE float8(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE float8& operator=(const T& rhs);

        //// INTEGER CASTING ////

        _CUDA_HD operator int8() const;

        _CUDA_HD operator uint8() const;

        _CUDA_HD operator int16() const;

        _CUDA_HD operator uint16() const;

        _CUDA_HD FORCEINLINE explicit operator int() const;

        _CUDA_HD FORCEINLINE explicit operator Nd4jLong() const;

        _CUDA_HD void assign(const int8& rhs);

        _CUDA_HD void assign(const uint8& rhs);

        _CUDA_HD void assign(const int16&  rhs);

        _CUDA_HD void assign(const uint16&  rhs);

        _CUDA_HD FORCEINLINE void assign(int rhs);

        _CUDA_HD FORCEINLINE void assign(Nd4jLong rhs);

        //// FLOAT CASTING ////

        _CUDA_HD operator float16() const;

        _CUDA_HD FORCEINLINE explicit operator float() const;

        _CUDA_HD FORCEINLINE explicit operator double() const;

        //// FLOAT ASSIGNING ////

        _CUDA_HD void assign(const float16& rhs);

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };


    float cpu_quarter2float(quarter b) {
        unsigned sign = ((b.x >> 7) & 1);
        unsigned exponent = ((b.x >> 4) & 0x7);
        unsigned mantissa = ((b.x & 0xf) << 19);

        if (exponent == 0x7) {  /* NaN or Inf */
            mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
            exponent = 0xff;
        } else if (!exponent) {  /* Denorm or Zero */
            if (mantissa) {
                unsigned int msb;
                exponent = 0x7d;
                do {
                    msb = (mantissa & 0x400000);
                    mantissa <<= 1;  /* normalize */
                    --exponent;
                } while (!msb);
                mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
            }
        } else {
            exponent += 0x7C;
        }

        int temp = ((sign << 31) | (exponent << 23) | mantissa);

        return *((float*)((void*)&temp));
    }



    quarter cpu_float2quarter_rn(float f)
    {
        quarter ret;

        unsigned x = *((int*)(void*)(&f));
        unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
        unsigned sign, exponent, mantissa;

        // Get rid of +NaN/-NaN case first.
        if (u > 0x7f800000) {
            ret.x = 0x7fU;
            return ret;
        }

        sign = ((x >> 24) & 0x80);

        // Get rid of +Inf/-Inf, +0/-0.
        if (u > 0x477fefff) {
            ret.x = sign | 0x70U;
            return ret;
        }
        if (u < 0x33000001) {
            ret.x = (sign | 0x00);
            return ret;
        }

        exponent = ((u >> 23) & 0xff);
        mantissa = (u & 0x7fffff);

        if (exponent > 0x7C) {
            shift = 19;
            exponent -= 0x7C;
        } else {
            shift = 0x90 - exponent;
            exponent = 0;
            mantissa |= 0x800000;
        }
        lsb = (1 << shift);
        lsb_s1 = (lsb >> 1);
        lsb_m1 = (lsb - 1);

        // Round to nearest even.
        remainder = (mantissa & lsb_m1);
        mantissa >>= shift;
        if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
            ++mantissa;
            if (!(mantissa & 0xf)) {
                ++exponent;
                mantissa = 0;
            }
        }

        ret.x = (sign | (exponent << 4) | mantissa);

        return ret;
    }

    float8::float8(quarter data) {
        this->data = data;
    }

    float8::float8() {
        data = cpu_float2quarter_rn(0.0f);
    }

    template <class T>
    float8& float8::operator=(const T& rhs) {
        assign(rhs); return *this;
    }

    template <class T>
    float8::float8(const T& rhs) {
        assign(rhs);
    }

    /////// CAST INT TYPES

    float8::operator int() const {
        return static_cast<int>(cpu_quarter2float(data));
    }

    float8::operator Nd4jLong() const {
        return static_cast<Nd4jLong>(cpu_quarter2float(data));
    }

    ///////  ASSIGN INT TYPES

    void float8::assign(int rhs) {
        assign((float)rhs);
    }

    void float8::assign(Nd4jLong rhs) {
        assign((float) rhs);
    }

    ///////  CAST FLOAT TYPES

    float8::operator float() const {
        return cpu_quarter2float(data);
    }

    float8::operator double() const {
        return static_cast<double>(cpu_quarter2float(data));
    }
    ///////  ASSIGN FLOAT TYPES

    void float8::assign(double rhs) {
        assign((float)rhs);
    }

    void float8::assign(float rhs) {
        data = cpu_float2quarter_rn(rhs);
    }
}

#endif //LIBND4J_FLOAT8_H
