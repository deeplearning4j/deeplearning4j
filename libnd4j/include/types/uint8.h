//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT8_H
#define LIBND4J_UINT8_H

#include <stdint.h>
#include <op_boilerplate.h>
#include <types/types.h>

namespace nd4j {

    float _CUDA_HD FORCEINLINE cpu_uint82float(uint8_t data);
    double _CUDA_HD FORCEINLINE cpu_uint82double(uint8_t data);
    uint8_t _CUDA_HD FORCEINLINE cpu_float2uint8(float data);

    struct uint8 {
        uint8_t data;

        _CUDA_HD FORCEINLINE uint8();
        _CUDA_HD FORCEINLINE ~uint8() = default;

        template <class T>
        _CUDA_HD FORCEINLINE uint8(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE uint8& operator=(const T& rhs);


        //// INTEGER CASTING ////

        _CUDA_HD operator int8() const;

        _CUDA_HD operator int16() const;

        _CUDA_HD operator uint16() const;

        _CUDA_HD FORCEINLINE explicit operator int() const;

        _CUDA_HD FORCEINLINE explicit operator Nd4jLong() const;

        //// INTEGER ASSIGNING ////

        _CUDA_HD FORCEINLINE void assign(uint8_t rhs);

        _CUDA_HD void assign(const int8& rhs);

        _CUDA_HD void assign(const int16&  rhs);

        _CUDA_HD void assign(const uint16&  rhs);

        _CUDA_HD FORCEINLINE void assign(int rhs);

        _CUDA_HD FORCEINLINE void assign(Nd4jLong rhs);

        //// FLOAT CASTING ////

        _CUDA_HD operator float16() const;

        _CUDA_HD operator float8() const;

        _CUDA_HD FORCEINLINE explicit operator float() const;

        _CUDA_HD FORCEINLINE explicit operator double() const;

        //// FLOAT ASSIGNING ////

        _CUDA_HD void assign(const float8& rhs);

        _CUDA_HD void assign(const float16& rhs);

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };



    ///////////////////////////


    float cpu_uint82float(uint8_t data) {
        return static_cast<float>(static_cast<int>(data));
    }

    double cpu_uint82double(uint8_t data) {
        return static_cast<double>(static_cast<int>(data));
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




    ///////  CAST INT TYPES

    uint8::operator int() const {
        return static_cast<int>(data);
    }

    uint8::operator Nd4jLong() const {
        return static_cast<Nd4jLong>(data);
    }

    ///////  ASSIGN INT TYPES
    
    void uint8::assign(uint8_t rhs) {
        data = rhs;
    }

    void uint8::assign(int rhs) {
        assign(static_cast<uint8_t>(rhs));
    }

    void uint8::assign(Nd4jLong rhs) {
        assign(static_cast<uint8_t>(rhs));
    }

    ///////  CAST FLOAT TYPES

    uint8::operator float() const {
        return cpu_uint82float(data);
    }

    uint8::operator double() const {
        return cpu_uint82double(data);
    }

    ///////  ASSIGN FLOAT TYPES

    void uint8::assign(double rhs) {
        assign(static_cast<float>(rhs));
    }

    void uint8::assign(float rhs) {
        data = cpu_float2uint8(rhs);
    }
}

#endif //LIBND4J_UINT8_H
