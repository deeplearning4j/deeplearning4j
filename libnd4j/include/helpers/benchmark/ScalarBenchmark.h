//
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

#ifndef DEV_TESTS_SCALARBENCHMARK_H
#define DEV_TESTS_SCALARBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT ScalarBenchmark : public OpBenchmark {
    public:
        ScalarBenchmark(scalar::Ops op, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
