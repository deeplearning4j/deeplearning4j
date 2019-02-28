//
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

namespace nd4j {
    OpBenchmark::OpBenchmark(NDArray *x, NDArray *y, NDArray *z) {
        _x = x;
        _y = y;
        _z = z;
    }

    OpBenchmark::OpBenchmark(NDArray *x, NDArray *z) {
        _x = x;
        _z = z;
    }

    OpBenchmark::OpBenchmark(NDArray *x, NDArray *z, std::initializer_list<int> axis) {
        _x = x;
        _z = z;
        _axis = axis;

        if (_axis.size() > 1)
            std::sort(_axis.begin(), _axis.end());
    }

    NDArray& OpBenchmark::x() {
        return *_x;
    }

    int OpBenchmark::opNum() {
        return _opNum;
    }


}