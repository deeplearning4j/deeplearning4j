/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
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

        ScalarBenchmark(scalar::Ops op, std::string testName) : OpBenchmark() {
            _opNum = (int) op;
            _testName = testName;
        }

        ScalarBenchmark(scalar::Ops op, std::string testName, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(testName, x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            if (_z == nullptr)
                NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
            else
                NativeOpExcutioner::execScalar(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), _y->buffer(), _y->shapeInfo(), nullptr);
        }

        std::string orders() override {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        std::string strides() override {
            std::string result;
            result += ShapeUtils::strideAsString(_x);
            result += "/";
            result += _z == nullptr ? ShapeUtils::strideAsString(_x) : ShapeUtils::strideAsString(_z);
            return result;
        }

        std::string axis() override {
            return "N/A";
        }

        std::string inplace() override {
            return _x == _z ? "true" : "false";
        }

        OpBenchmark* clone() override  {
            return new ScalarBenchmark((scalar::Ops) _opNum, _testName, _x == nullptr ? _x : _x->dup() , _y == nullptr ? _y : _y->dup(), _z == nullptr ? _z : _z->dup());
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H