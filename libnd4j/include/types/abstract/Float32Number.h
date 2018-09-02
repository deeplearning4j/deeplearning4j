//
// Created by raver on 9/2/2018.
//

#ifndef LIBND4J_FLOAT32NUMBER_H
#define LIBND4J_FLOAT32NUMBER_H

#include <types/abstract/Number.h>

namespace nd4j {
    class Float32Number : public Number  {
    protected:
        float _value = 0.0f;
    public:
        virtual double asDoubleValue();
        virtual float asFloatValue();
        virtual float16 asHalfValue();
        virtual int asInt32Value();
        virtual int16_t asInt16Value();
        virtual int8_t asInt8Value();
        virtual bool asBoolValue();
    };
}


#endif //DEV_TESTS_FLOAT32NUMBER_H
