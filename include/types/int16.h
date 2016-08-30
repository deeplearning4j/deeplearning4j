//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INT16_H
#define LIBND4J_INT16_H

#include <stdint.h>


float cpu_int162float(int16_t data) {
    return (float) ((int) data);
}

int16_t cpu_float2int16(float data) {
    int t = (int) data;
    if (t > 32767 ) t = 32767;
    if (t < -32768) t = -32768;

    return (int16_t) t;
}


namespace nd4j {

    struct int16 {
        int16_t data;

        local_def int16() { data = cpu_float2int16(0.0f); }

        template <class T>
        local_def int16(const T& rhs) {
            assign(rhs);
        }

        template <class T>
        local_def int16& operator=(const T& rhs) { assign(rhs); return *this; }


        local_def operator float() const {
            return cpu_int162float(data);
        }

        local_def void assign(double rhs) {
            assign((float)rhs);
        }

        local_def void assign(float rhs) {
            data = cpu_float2int16(rhs);
        }
    };
}

#endif //LIBND4J_INT16_H
