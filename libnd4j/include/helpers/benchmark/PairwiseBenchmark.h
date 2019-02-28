//
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

#ifndef DEV_TESTS_PAIRWISEBENCHMARK_H
#define DEV_TESTS_PAIRWISEBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT PairwiseBenchmark : public OpBenchmark {
    public:
        PairwiseBenchmark() : OpBenchmark() {
            //
        }

        PairwiseBenchmark(pairwise::Ops op, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execPairwiseTransform(_opNum, _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr);
        }

        OpBenchmark* clone() override  {
            return new PairwiseBenchmark((pairwise::Ops) _opNum, _x, _y, _z);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
