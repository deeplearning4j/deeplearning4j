//
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

#ifndef DEV_TESTS_SCALARBENCHMARK_H
#define DEV_TESTS_SCALARBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT ScalarBenchmark : public OpBenchmark {
    public:
        ScalarBenchmark() : OpBenchmark() {
            //
        }

        ~ScalarBenchmark(){
            if (_x != _y && _x != _z && _y != _z) {
                delete _x;
                delete _y;
                delete _z;
            } else if (_x == _y && _x == _z) {
                delete _x;
            } else if (_x == _z) {
                delete _x;
                delete _y;
            }
        }

        ScalarBenchmark(scalar::Ops op) : OpBenchmark() {
            _opNum = (int) op;
        }

        ScalarBenchmark(scalar::Ops op, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            if (_z == nullptr)
                NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
            else
                NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
        }

        std::string orders() {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        OpBenchmark* clone() override  {
            return new ScalarBenchmark((scalar::Ops) _opNum, _x == nullptr ? _x : _x->dup() , _y == nullptr ? _y : _y->dup(), _z == nullptr ? _z : _z->dup());
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
