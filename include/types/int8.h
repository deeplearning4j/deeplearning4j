//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INT8_H
#define LIBND4J_INT8_H

#include <stdint.h>


float cpu_int82float(int8_t data) {
    return (float) ((int) data);
}

int8_t cpu_float2int8(float data) {
    int t = (int) data;
    if (t > 127) t = 127;
    if (t < -128) t = -128;

    return (int8_t) t;
}


namespace nd4j {

    struct int8 {
        int8_t data;

        local_def int8() { data = cpu_float2int8(0.0f); }

        template <class T>
        local_def int8(const T& rhs) {
            assign(rhs);
        }

        template <class T>
        local_def int8& operator=(const T& rhs) { assign(rhs); return *this; }


        local_def operator float() const {
            return cpu_int82float(data);
        }

        local_def void assign(double rhs) {
            assign((float)rhs);
        }

        local_def void assign(float rhs) {
            data = cpu_float2int8(rhs);
        }
    };
}

#endif //LIBND4J_INT8_H
