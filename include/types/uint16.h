//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT16_H
#define LIBND4J_UINT16_H

#include <stdint.h>


float cpu_uint162float(int16_t data) {
    return (float) ((int) data);
}

uint8_t cpu_float2uint16(float data) {
    int t = (int) data;
    if (t > 65536 ) t = 65536;
    if (t < 0) t = 0;

    return (uint16_t) t;
}


namespace nd4j {

    struct uint16 {
        uint16_t data;

        local_def uint16() { data = cpu_float2uint16(0.0f); }

        template <class T>
        local_def uint16(const T& rhs) {
            assign(rhs);
        }

        template <class T>
        local_def uint16& operator=(const T& rhs) { assign(rhs); return *this; }


        local_def operator float() const {
            return cpu_uint162float(data);
        }

        local_def void assign(double rhs) {
            assign((float)rhs);
        }

        local_def void assign(float rhs) {
            data = cpu_float2uint16(rhs);
        }
    };
}

#endif //LIBND4J_UINT16_H
