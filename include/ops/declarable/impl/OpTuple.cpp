//
// Created by raver119 on 11.10.2017.
//

#include "ops/declarable/OpTuple.h"

nd4j::ops::OpTuple::OpTuple(const char *opName, std::initializer_list<nd4j::NDArray<float> *> &&inputs, std::initializer_list<float> &&tArgs, std::initializer_list<int> &&iArgs) {
    _opName = opName;
    _inputs = inputs;
    _iArgs = iArgs;
    _tArgs = tArgs;
}
