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
        NDArray *_x;
        NDArray *_y;
        NDArray *_z;
        std::vector<int> _axis;
    public:
        OpBenchmark(NDArray *x, NDArray *y, NDArray *z) {
            _x = x;
            _y = y;
            _z = z;
        }

        NDArray& x() {
            return *_x;
        }

        int opNum() {
            return _opNum;
        }

        OpBenchmark(NDArray *x, NDArray *z) {
            _x = x;
            _z = z;
        }

        OpBenchmark(NDArray *x, NDArray *z, std::initializer_list<int> axis) {
            _x = x;
            _z = z;
            _axis = axis;

            if (_axis.size() > 1)
                std::sort(_axis.begin(), _axis.end());
        }

        virtual void executeOnce();
    };


    class ScalarBenchmark : OpBenchmark {
    public:
        ScalarBenchmark(scalar::Ops op, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
        }
    };

    class PairwiseBenchmark : OpBenchmark {
    public:
        PairwiseBenchmark(pairwise::Ops op, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execPairwiseTransform(_opNum, _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr);
        }
    };

    class TransformBenchmark : OpBenchmark {
    public:
        TransformBenchmark(transform::StrictOps op, NDArray *x, NDArray *z) : OpBenchmark(x, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execTransformStrict(_opNum, _x->buffer(), _x->shapeInfo(),  _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
        }
    };

    class ReductionBenchmark : OpBenchmark {
    public:
        ReductionBenchmark(reduce::FloatOps op, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(x, z, axis) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            NativeOpExcutioner::execReduceFloat(_opNum, _x->buffer(), _x->shapeInfo(), nullptr,  _z->buffer(), _z->shapeInfo(), _axis.data(), _axis.size(), nullptr, nullptr);
        }
    };
}


#endif //DEV_TESTS_OPEXECUTIONER_H
