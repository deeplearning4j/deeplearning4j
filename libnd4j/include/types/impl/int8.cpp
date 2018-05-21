//
// Created by raver119 on 07/11/17.
//

#include <types/int8.h>

namespace nd4j {
    float cpu_int82float(int8_t data) {
        return (float) ((int) data);
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


    int8::operator float() const {
        return cpu_int82float(data);
    }

    void int8::assign(double rhs) {
        assign((float)rhs);
    }

    void int8::assign(float rhs) {
        data = cpu_float2int8(rhs);
    }

    template int8::int8(const float& rhs);
    template int8::int8(const double& rhs);

    template int8& int8::operator=<float>(const float& rhs);
    template int8& int8::operator=<double>(const double& rhs);
}