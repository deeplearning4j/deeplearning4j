//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT8_H
#define LIBND4J_UINT8_H

#include <stdint.h>


float cpu_uint82float(uint8_t data) {
    return (float) ((int) data);
}

uint8_t cpu_float2uint8(float data) {
    int t = (int) data;
    if (t > 255) t = 255;
    if (t < 0) t = 0;

    return (uint8_t) t;
}


namespace nd4j {

    struct uint8 {
        uint8_t data;

        local_def uint8() { data = cpu_float2uint8(0.0f); }

        template <class T>
        local_def uint8(const T& rhs) {
            assign(rhs);
        }

        template <class T>
        local_def uint8& operator=(const T& rhs) { assign(rhs); return *this; }


        local_def operator float() const {
            return cpu_uint82float(data);
        }

        local_def void assign(double rhs) {
            assign((float)rhs);
        }

        local_def void assign(float rhs) {
            data = cpu_float2uint8(rhs);
        }
    };
}

#endif //LIBND4J_UINT8_H
