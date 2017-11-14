//
// Created by raver119 on 07/11/17.
//

#include <types/int16.h>

namespace nd4j {
    float cpu_int162float(int16_t data) {
        return (float) ((int) data);
    }

    int16_t cpu_float2int16(float data) {
        int t = (int) data;
        if (t > 32767 ) t = 32767;
        if (t < -32768) t = -32768;

        return (int16_t) t;
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


    int16::operator float() const {
        return cpu_int162float(data);
    }

    void int16::assign(double rhs) {
        assign((float)rhs);
    }

    void int16::assign(float rhs) {
        data = cpu_float2int16(rhs);
    }

    template int16::int16(const float& rhs);
    template int16::int16(const double& rhs);

    template int16& int16::operator=<float>(const float& rhs);
    template int16& int16::operator=<double>(const double& rhs);
}