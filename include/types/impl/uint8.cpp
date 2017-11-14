//
// Created by raver119 on 07/11/17.
//

#include <types/uint8.h>

namespace nd4j {

    float cpu_uint82float(uint8_t data) {
        return (float) ((int) data);
    }

    uint8_t cpu_float2uint8(float data) {
        int t = (int) data;
        if (t > 255) t = 255;
        if (t < 0) t = 0;

        return (uint8_t) t;
    }

    uint8::uint8() { data = cpu_float2uint8(0.0f); }

    template <class T>
    uint8::uint8(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    uint8& uint8::operator=(const T& rhs) { assign(rhs); return *this; }


    uint8::operator float() const {
        return cpu_uint82float(data);
    }

    void uint8::assign(double rhs) {
        assign((float)rhs);
    }

    void uint8::assign(float rhs) {
        data = cpu_float2uint8(rhs);
    }

    template uint8::uint8(const float& rhs);
    template uint8::uint8(const double& rhs);

    template uint8& uint8::operator=<float>(const float& rhs);
    template uint8& uint8::operator=<double>(const double& rhs);
}