//
// Created by raver on 4/5/2018.
//

#include <types/triple.h>

namespace nd4j {
    int Triple::first() const {
        return _first;
    }

    int Triple::second() const {
        return _second;
    }

    int Triple::third() const {
        return _third;
    }

    Triple::Triple(int first, int second, int third) {
        _first = first;
        _second = second;
        _third = third;
    }
}
