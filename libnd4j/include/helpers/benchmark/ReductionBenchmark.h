//
// Created by raver on 2/28/2019.
//

#include <helpers/StringUtils.h>
#include "../OpBenchmark.h"

#ifndef DEV_TESTS_REDUCEBENCHMARK_H
#define DEV_TESTS_REDUCEBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT ReductionBenchmark : public OpBenchmark {
    public:
        ReductionBenchmark() : OpBenchmark() {
            //
        }

        ReductionBenchmark(reduce::FloatOps op, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(x, z, axis) {
            _opNum = (int) op;
        }

        ReductionBenchmark(reduce::FloatOps op) : OpBenchmark() {
            _opNum = (int) op;
        }

        ReductionBenchmark(reduce::FloatOps op, NDArray *x, NDArray *z, std::vector<int> axis) : OpBenchmark(x, z, axis) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            if (_z->isScalar())
                NativeOpExcutioner::execReduceFloatScalar(_opNum, _x->buffer(), _x->shapeInfo(), nullptr, _z->buffer(), _z->shapeInfo());
            else
                NativeOpExcutioner::execReduceFloat(_opNum, _x->buffer(), _x->shapeInfo(), nullptr,  _z->buffer(), _z->shapeInfo(), _axis.data(), _axis.size(), nullptr, nullptr);
        }

        std::string orders() {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        ~ReductionBenchmark(){
            delete _x;
            delete _z;
        }

        std::string axis() {
            if (_axis.empty())
                return "ALL";
            else {
                std::string result;
                for (auto v:_axis) {
                    auto s = StringUtils::valueToString<int>(v);
                    result += s;
                    result += ",";
                }

                return result;
            }
        }

        OpBenchmark* clone() override  {
            return new ReductionBenchmark((reduce::FloatOps) _opNum, _x, _z, _axis);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
