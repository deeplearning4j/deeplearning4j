/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Alex Black
//

#include "../OpBenchmark.h"

#ifndef DEV_TESTS_BROADCASTBENCHMARK_H
#define DEV_TESTS_BROADCASTBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT BroadcastBenchmark : public OpBenchmark {
    public:
        BroadcastBenchmark() : OpBenchmark() {
            //
        }

        BroadcastBenchmark(broadcast::Ops op, std::string testName, NDArray *x, NDArray *y, NDArray *z, std::vector<int> axis) : OpBenchmark(testName, x, y, z, axis) {
            _opNum = (int) op;
        }

        BroadcastBenchmark(broadcast::Ops op, std::string testName, NDArray *x, NDArray *y, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(testName, x, y, z, axis) {
            _opNum = (int) op;
        }

        BroadcastBenchmark(broadcast::Ops op, std::string name, std::vector<int> axis) : OpBenchmark() {
            _opNum = (int) op;
            _testName = name;
            _axis = axis;
        }

        BroadcastBenchmark(broadcast::Ops op, std::string name, std::initializer_list<int> axis) : OpBenchmark() {
            _opNum = (int) op;
            _testName = name;
            _axis = axis;
        }

        ~BroadcastBenchmark(){
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
            //TODO: TAD pointers
            NativeOpExcutioner::execBroadcast(_opNum, _x->buffer(), _x->shapeInfo(), _y->buffer(), _y->shapeInfo(), _z->buffer(), _z->shapeInfo(),
                _axis.data(), _axis.size(), /*Nd4jLong *tadOnlyShapeInfo*/ nullptr, /*Nd4jLong *tadOffsets*/ nullptr,
                /*Nd4jLong *tadOnlyShapeInfoZ*/ nullptr, /*Nd4jLong *tadOffsetsZ*/ nullptr);
        }

        std::string axis() override {
            if (_axis.empty())
                return "<none>";
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

        std::string inplace() override {
            std::string result;
            result += (_x == _z ? "true" : "false");
            return result;
        }

        std::string orders() override {
            std::string result;
            result += _x->ordering();
            result += "/";
            result += _y->ordering();
            result += "/";
            result += _z == nullptr ? _x->ordering() : _z->ordering();
            return result;
        }

        std::string strides() override {
            std::string result;
            result += ShapeUtils::strideAsString(_x);
            result += "/";
            result += ShapeUtils::strideAsString(_y);
            result += "/";
            result += _z == nullptr ? ShapeUtils::strideAsString(_x) : ShapeUtils::strideAsString(_z);
            return result;
        }

        OpBenchmark* clone() override  {
            return new BroadcastBenchmark((broadcast::Ops) _opNum, _testName, _x, _y, _z, _axis);
        }
    };
}

#endif //DEV_TESTS_BROADCASTBENCHMARK_H