//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT8_H
#define LIBND4J_UINT8_H

#include <stdint.h>
#include <op_boilerplate.h>


namespace nd4j {

    float _CUDA_HD FORCEINLINE cpu_uint82float(uint8_t data);
    uint8_t _CUDA_HD FORCEINLINE cpu_float2uint8(float data);

    struct uint8 {
        uint8_t data;

        _CUDA_HD FORCEINLINE uint8();
        _CUDA_HD FORCEINLINE ~uint8() = default;

        template <class T>
        _CUDA_HD FORCEINLINE uint8(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE uint8& operator=(const T& rhs);


        _CUDA_HD FORCEINLINE operator float() const;

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };



    ///////////////////////////


    float cpu_uint82float(uint8_t data) {
        return static_cast<float>(static_cast<int>(data));
    }

    uint8_t cpu_float2uint8(float data) {
        auto t = static_cast<int>(data);
        if (t > 255) t = 255;
        if (t < 0) t = 0;

        return static_cast<uint8_t>(t);
    }

    uint8::uint8() { data = cpu_float2uint8(0.0f); }

    template <class T>
    uint8::uint8(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    uint8& uint8::operator=(const T& rhs) { assign(rhs); return *this; }


    uint8::operator float() const {
        return cpu_uint82float(data);
    }

    void uint8::assign(double rhs) {
        assign(static_cast<float>(rhs));
    }

    void uint8::assign(float rhs) {
        data = cpu_float2uint8(rhs);
    }
}

#endif //LIBND4J_UINT8_H
