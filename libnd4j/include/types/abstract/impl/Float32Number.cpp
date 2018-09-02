//
// Created by raver on 9/2/2018.
//

#include <types/abstract/Float32Number.h>

namespace nd4j {
    double Float32Number::asDoubleValue() {
        return static_cast<double>(_value);
    }

    float Float32Number::asFloatValue() {
        return _value;
    }

    float16 Float32Number::asHalfValue() {
        return static_cast<float16>(_value);
    }

    int Float32Number::asInt32Value() {
        return static_cast<int>(_value);
    }

    int16_t Float32Number::asInt16Value() {
        return static_cast<int16_t>(_value);
    }

    int8_t Float32Number::asInt8Value() {
        return static_cast<int8_t>(_value);
    }

    bool Float32Number::asBoolValue() {
        return _value == 0.0f;
    }
}
