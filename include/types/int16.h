//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INT16_H
#define LIBND4J_INT16_H

#include <stdint.h>


namespace nd4j {

    float cpu_int162float(int16_t data);
    int16_t cpu_float2int16(float data);

    struct int16 {
        int16_t data;

        int16();
        ~int16() = default;

        template <class T>
        int16(const T& rhs);

        template <class T>
        int16& operator=(const T& rhs);


        operator float() const;

        void assign(double rhs);

        void assign(float rhs);
    };
}

#endif //LIBND4J_INT16_H
