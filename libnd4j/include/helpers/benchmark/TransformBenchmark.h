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

#ifndef DEV_TESTS_TRANSFORMBENCHMARK_H
#define DEV_TESTS_TRANSFORMBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT TransformBenchmark : public OpBenchmark {

    protected:
        int _opType;        // 0=StrictOps, 1=Same, 2=Any, 3=Float

    public:
        TransformBenchmark() : OpBenchmark() {
            //
        }

        TransformBenchmark(int opNum, int opType, std::string testName, NDArray *x, NDArray *z) : OpBenchmark(testName, x, z) {
            _opNum = opNum;
            _opType = opType;
        }

        TransformBenchmark(transform::StrictOps op, std::string testName, NDArray *x, NDArray *z) : OpBenchmark(testName, x, z) {
            _opNum = (int) op;
            _opType = 0;
        }

        TransformBenchmark(transform::StrictOps op, std::string name) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 0;
            _testName = name;
        }

        TransformBenchmark(transform::SameOps op, std::string name) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 1;
            _testName = name;
        }

        TransformBenchmark(transform::AnyOps op, std::string name) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 2;
            _testName = name;
        }

        TransformBenchmark(transform::FloatOps op, std::string name) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 3;
            _testName = name;
        }

        ~TransformBenchmark(){

            if (_x == _z) {
                delete _x;
            } else {
                delete _x;
                delete _z;
            }
        }

        void executeOnce() override {
            if(_opType == 0){
                if (_z != nullptr)
                    NativeOpExcutioner::execTransformStrict(_opNum, _x->buffer(), _x->shapeInfo(),  _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                else
                    NativeOpExcutioner::execTransformStrict(_opNum, _x->buffer(), _x->shapeInfo(),  _x->buffer(), _x->shapeInfo(), nullptr, nullptr, nullptr);
            } else if(_opType == 1){
                if (_z != nullptr){
                    NativeOpExcutioner::execTransformSame(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                } else {
                    NativeOpExcutioner::execTransformSame(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                }
            } else if(_opType == 2){
                if (_z != nullptr){
                    NativeOpExcutioner::execTransformAny(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                } else {
                    NativeOpExcutioner::execTransformAny(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                }
            } else {
                if (_z != nullptr){
                    NativeOpExcutioner::execTransformFloat(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                } else {
                    NativeOpExcutioner::execTransformFloat(_opNum, _x->buffer(), _x->shapeInfo(), _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
                }
            }
        }

        std::string axis() override {
            return "N/A";
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

        std::string inplace() override {
            return _x == _z ? "true" : "false";
        }

        OpBenchmark* clone() override  {
            return new TransformBenchmark(_opNum, _opType, _testName, _x, _z);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H