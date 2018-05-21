//
// Created by raver119 on 07/11/17.
//

#include <types/float8.h>

namespace nd4j {
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


    float8::float8() {
        data = cpu_float2quarter_rn(0.0f);
    }

    template <class T>
    float8::float8(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    float8& float8::operator=(const T& rhs) {
        assign(rhs); return *this;
    }


    float8::operator float() const {
        return cpu_quarter2float(data);
    }

    void float8::assign(double rhs) {
        assign((float)rhs);
    }

    void float8::assign(float rhs) {
        data = cpu_float2quarter_rn(rhs);
    }

    template float8::float8(const float& rhs);
    template float8::float8(const double& rhs);

    template float8& float8::operator=<float>(const float& rhs);
    template float8& float8::operator=<double>(const double& rhs);
}