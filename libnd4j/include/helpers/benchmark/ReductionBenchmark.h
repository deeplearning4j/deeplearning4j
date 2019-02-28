//
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

#ifndef DEV_TESTS_REDUCEBENCHMARK_H
#define DEV_TESTS_REDUCEBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT ReductionBenchmark : public OpBenchmark {
    public:
        ReductionBenchmark(reduce::FloatOps op, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(x, z, axis) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execReduceFloat(_opNum, _x->buffer(), _x->shapeInfo(), nullptr,  _z->buffer(), _z->shapeInfo(), _axis.data(), _axis.size(), nullptr, nullptr);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
