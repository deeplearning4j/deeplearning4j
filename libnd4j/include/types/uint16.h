//
// @author raver119@gmail.com
//

#ifndef LIBND4J_UINT16_H
#define LIBND4J_UINT16_H

#include <stdint.h>
#include <op_boilerplate.h>


namespace nd4j {

    uint16_t _CUDA_HD FORCEINLINE cpu_float2uint16(float data);
    float _CUDA_HD FORCEINLINE cpu_uint162float(uint16_t data);
    double _CUDA_HD FORCEINLINE cpu_uint162double(uint16_t data);

    struct uint16 {
        uint16_t data;

        _CUDA_HD FORCEINLINE uint16();
        _CUDA_HD FORCEINLINE ~uint16();

        template <class T>
        _CUDA_HD FORCEINLINE uint16(const T& rhs);

        template <class T>
        _CUDA_HD FORCEINLINE uint16& operator=(const T& rhs);

        //// INTEGER CASTING ////

        _CUDA_HD FORCEINLINE explicit operator int() const;

        _CUDA_HD FORCEINLINE explicit operator Nd4jLong() const;

        //// INTEGER ASSIGNING ////
        _CUDA_HD FORCEINLINE void assign(uint16_t rhs);

        _CUDA_HD FORCEINLINE void assign(int rhs);

        _CUDA_HD FORCEINLINE void assign(Nd4jLong rhs);

        //// FLOAT CASTING ////

        _CUDA_HD FORCEINLINE explicit operator float() const;

        _CUDA_HD FORCEINLINE explicit operator double() const;

        //// FLOAT ASSIGNING ////

        _CUDA_HD FORCEINLINE void assign(double rhs);

        _CUDA_HD FORCEINLINE void assign(float rhs);
    };

//////////////////// IMPLEMENTATIONS

    float _CUDA_HD cpu_uint162float(uint16_t data) {
        return static_cast<float>(data);
    }

    double _CUDA_HD cpu_uint162double(uint16_t data) {
        return static_cast<double>(data);
    }
    
    uint16_t _CUDA_HD cpu_float2uint16(float data) {
        auto t = static_cast<int>(data);
        if (t > 65536 ) t = 65536;
        if (t < 0) t = 0;

        return static_cast<uint16_t>(t);
    }

    _CUDA_HD uint16::uint16() {
        data = cpu_float2uint16(0.0f);
    }

    _CUDA_HD uint16::~uint16() {
        //
    }

    template <class T>
    _CUDA_HD uint16::uint16(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    _CUDA_HD uint16& uint16::operator=(const T& rhs) {
        assign(rhs);
        return *this;
    }
    
    ///////  CAST INT TYPES
    uint16::operator int() const {
        return static_cast<int>(data);
    }

    uint16::operator Nd4jLong() const {
        return static_cast<Nd4jLong>(data);
    }

    ///////  ASSIGN INT TYPES
    
    _CUDA_HD void uint16::assign(uint16_t rhs) {
        data = rhs;
    }


    void uint16::assign(int rhs) {
        assign(static_cast<int8_t>(rhs));
    }

    void uint16::assign(Nd4jLong rhs) {
        assign(static_cast<int8_t>(rhs));
    }

    ///////  CAST FLOAT TYPES
    
    _CUDA_HD uint16::operator float() const {
        return cpu_uint162float(data);
    }

    _CUDA_HD uint16::operator double() const {
        return cpu_uint162double(data);
    }

    ///////  ASSIGN FLOAT TYPES

    _CUDA_HD void uint16::assign(float rhs) {
        data = cpu_float2uint16(rhs);
    }

    _CUDA_HD void uint16::assign(double rhs) {
        assign((float)rhs);
    }
}

#endif //LIBND4J_UINT16_H
