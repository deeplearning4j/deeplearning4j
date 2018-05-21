//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT16_H
#define LIBND4J_UINT16_H

#include <stdint.h>


namespace nd4j {

    uint8_t cpu_float2uint16(float data);
    float cpu_uint162float(int16_t data);

    struct uint16 {
        uint16_t data;

        uint16();
        ~uint16();

        template <class T>
        uint16(const T& rhs);

        template <class T>
        uint16& operator=(const T& rhs);

        operator float() const;

        void assign(double rhs);

        void assign(float rhs);
    };
}

#endif //LIBND4J_UINT16_H
