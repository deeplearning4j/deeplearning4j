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

        PairwiseBenchmark(pairwise::Ops op, std::string testName, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(testName, x, y, z) {
            _opNum = (int) op;
        }

        PairwiseBenchmark(pairwise::Ops op) : OpBenchmark() {
            _opNum = (int) op;
        }

        ~PairwiseBenchmark(){
            if (_x != _y && _x != _z && _y != _z) {
                delete _x;
                delete _y;
                delete _z;
            } else if (_x == _y && _x == _z) {
                delete _x;
            } else if (_x == _z) {
                delete _x;
                delete _y;
            } else if (_y == _z) {
                delete _x;
                delete _y;
            }
        }

        void executeOnce() override {
            NativeOpExcutioner::execPairwiseTransform(_opNum, _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr);
        }

        std::string axis() {
            return "N/A";
        }

        std::string orders() {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _y->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        OpBenchmark* clone() override  {
            return new PairwiseBenchmark((pairwise::Ops) _opNum, _testName, _x, _y, _z);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
