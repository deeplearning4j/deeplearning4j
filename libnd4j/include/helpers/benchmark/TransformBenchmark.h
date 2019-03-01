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
    public:
        TransformBenchmark() : OpBenchmark() {
            //
        }

        TransformBenchmark(transform::StrictOps op, std::string testName, NDArray *x, NDArray *z) : OpBenchmark(testName, x, z) {
            _opNum = (int) op;
        }

        TransformBenchmark(transform::StrictOps op) : OpBenchmark() {
            _opNum = (int) op;
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
            if (_z != nullptr)
                NativeOpExcutioner::execTransformStrict(_opNum, _x->buffer(), _x->shapeInfo(),  _z->buffer(), _z->shapeInfo(), nullptr, nullptr, nullptr);
            else
                NativeOpExcutioner::execTransformStrict(_opNum, _x->buffer(), _x->shapeInfo(),  _x->buffer(), _x->shapeInfo(), nullptr, nullptr, nullptr);
        }

        std::string axis() {
            return "N/A";
        }

        std::string orders() {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        OpBenchmark* clone() override  {
            return new TransformBenchmark((transform::StrictOps) _opNum, _testName, _x, _z);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
