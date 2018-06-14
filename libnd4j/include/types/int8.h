//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INT8_H
#define LIBND4J_INT8_H

#include <stdint.h>
#include <op_boilerplate.h>


namespace nd4j {

    float _CUDA_HD FORCEINLINE cpu_int82float(int8_t data);
    int8_t _CUDA_HD FORCEINLINE cpu_float2int8(float data);

    struct int8 {
        int8_t data;

        _CUDA_HD FORCEINLINE int8();
        _CUDA_HD FORCEINLINE ~int8() = default;

        template <class T>
        _CUDA_HD FORCEINLINE int8(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE int8& operator=(const T& rhs);


        _CUDA_HD FORCEINLINE operator float() const;

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };


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
}

#endif //LIBND4J_INT8_H
