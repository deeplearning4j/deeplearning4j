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

#ifndef DEV_TESTS_PAIRWISEBENCHMARK_H
#define DEV_TESTS_PAIRWISEBENCHMARK_H

using namespace nd4j::graph;

namespace nd4j {
    class ND4J_EXPORT PairwiseBenchmark : public OpBenchmark {
    public:
        PairwiseBenchmark() : OpBenchmark() {
            //
        }

        PairwiseBenchmark(pairwise::Ops op, std::string testName, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(testName, x, y, z) {
            _opNum = (int) op;
        }

        PairwiseBenchmark(pairwise::Ops op, std::string name) : OpBenchmark() {
            _opNum = (int) op;
            _testName = name;
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
            PointersManager manager(LaunchContext::defaultContext(), "PairwiseBM");

            NativeOpExecutioner::execPairwiseTransform(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), _y->buffer(), _y->shapeInfo(), _y->specialBuffer(), _y->specialShapeInfo(), _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo(), nullptr);

            manager.synchronize();
        }

        std::string axis() override {
            return "N/A";
        }

        std::string inplace() override {
            std::string result;
            result += (_x == _y ? "x==y" : "x!=y");
            result += "/";
            result += (_x == _z ? "x==z" : "x!=z");
            result += "/";
            result += (_y == _z ? "y==z" : "y!=z");
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
            return new PairwiseBenchmark((pairwise::Ops) _opNum, _testName, _x, _y, _z);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H