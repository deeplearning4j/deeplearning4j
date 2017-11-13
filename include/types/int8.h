//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INT8_H
#define LIBND4J_INT8_H

#include <stdint.h>


namespace nd4j {

    float cpu_int82float(int8_t data);
    int8_t cpu_float2int8(float data);

    struct int8 {
        int8_t data;

        int8();
        ~int8() = default;

        template <class T>
        int8(const T& rhs);

        template <class T>
        int8& operator=(const T& rhs);


        operator float() const;

        void assign(double rhs);

        void assign(float rhs);
    };
}

#endif //LIBND4J_INT8_H
