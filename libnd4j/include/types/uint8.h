//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT8_H
#define LIBND4J_UINT8_H

#include <stdint.h>


namespace nd4j {

    float cpu_uint82float(uint8_t data);
    uint8_t cpu_float2uint8(float data);

    struct uint8 {
        uint8_t data;

        uint8();
        ~uint8() = default;

        template <class T>
        uint8(const T& rhs);

        template <class T>
        uint8& operator=(const T& rhs);


        operator float() const;

        void assign(double rhs);

        void assign(float rhs);
    };
}

#endif //LIBND4J_UINT8_H
