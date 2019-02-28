//
// Created by raver on 2/28/2019.
//

#ifndef DEV_TESTS_OPEXECUTIONER_H
#define DEV_TESTS_OPEXECUTIONER_H

#include <NativeOpExcutioner.h>
#include <NDArray.h>

namespace nd4j {
    class OpBenchmark {
    protected:
        int _opNum;
    public:
        virtual void executeOnce(NDArray *x, NDArray *y, NDArray *z);
    };


    class ScalarBenchmark : OpBenchmark {
    public:
        ScalarBenchmark(scalar::Ops op) : OpBenchmark() {
            _opNum = (int) op;
        }

        void executeOnce(NDArray *x, NDArray *y, NDArray *z) override {
            NativeOpExcutioner::execScalar(_opNum, x->buffer(), x->shapeInfo(), z->buffer(), z->shapeInfo(), y->buffer(), y->shapeInfo(), nullptr);
        }
    };

    class PairwiseBenchmark : OpBenchmark {
    public:
        PairwiseBenchmark(pairwise::Ops op) : OpBenchmark() {
            _opNum = (int) op;
        }

        void executeOnce(NDArray *x, NDArray *y, NDArray *z) override {
            NativeOpExcutioner::execPairwiseTransform(_opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), nullptr);
        }
    };
}


#endif //DEV_TESTS_OPEXECUTIONER_H
