//
// Created by raver119 on 07/11/17.
//
#include <types/float8.h>
#include <types/float16.h>
#include <types/int8.h>
#include <types/uint8.h>
#include <types/int16.h>
#include <types/uint16.h>

namespace nd4j {
    ///////  CAST INT TYPES

    int16::operator int8() const {
        return int8(static_cast<int8_t>(data));
    }

    int16::operator uint8() const {
        return uint8(static_cast<uint8_t>(data));
    }


    int16::operator uint16() const {
        return uint16(static_cast<uint16_t>(data));
    }

    ///////  ASSIGN INT TYPES

    void int16::assign(const int8& rhs) {
        assign(static_cast<int16_t>(rhs.data));
    }

    void int16::assign(const uint8& rhs) {
        assign(static_cast<int16_t>(rhs.data));
    }


    void int16::assign(const uint16& rhs) {
        assign(static_cast<int16_t>(rhs.data));
    }

    ///////  CAST CUSTOM FLOAT TYPES
    int16::operator float8() const {
        return static_cast<float8>(cpu_int162float(data));
    }

    int16::operator float16() const {
        return static_cast<float16>(cpu_int162float(data));
    }

    ///////  ASSIGN CUSTOM FLOAT TYPES
    void int16::assign(const float8& rhs) {
        assign((float)rhs);
    }

    void int16::assign(const float16& rhs) {
        assign((float)rhs);
    }
}