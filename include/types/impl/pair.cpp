//
// Created by raver119 on 24.01.18.
//

#include <types/pair.h>

namespace nd4j {
    Pair::Pair(int first, int second) {
        _first = first;
        _second = second;
    }

    int Pair::first() const {
        return _first;
    }

    int Pair::second() const {
        return _second;
    };
}
