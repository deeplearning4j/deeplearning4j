//
// Created by agibsonccc on 1/28/16.
//

#include "NativeOpExcutioner.h"

class DoubleNativeOpExecutioner : public NativeOpExcutioner<double> {
private:
    DoubleNativeOpExecutioner *INSTANCE;
public:
    DoubleNativeOpExecutioner getInstance() {
        if(INSTANCE == NULL)
            INSTANCE = new DoubleNativeOpExecutioner();
        return INSTANCE;
    }
};

class FloatNativeOpExecutioner : public NativeOpExcutioner<float> {
private:
    FloatNativeOpExecutioner *INSTANCE;
public:
    FloatNativeOpExecutioner getInstance() {
        if(INSTANCE == NULL)
            INSTANCE = new FloatNativeOpExecutioner();
        return INSTANCE;
    }
};
