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

#include <helpers/StringUtils.h>
#include <helpers/TAD.h>
#include "../OpBenchmark.h"

#ifndef DEV_TESTS_REDUCEBENCHMARK_H
#define DEV_TESTS_REDUCEBENCHMARK_H

using namespace nd4j::graph;

namespace nd4j {
    class ND4J_EXPORT ReductionBenchmark : public OpBenchmark {
    public:
        ReductionBenchmark() : OpBenchmark() {
            //
        }

        ReductionBenchmark(reduce::FloatOps op, std::string testName, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(testName, x, z, axis) {
            _opNum = (int) op;
        }

        ReductionBenchmark(reduce::FloatOps op) : OpBenchmark() {
            _opNum = (int) op;
        }

        ReductionBenchmark(reduce::FloatOps op, std::string testName, NDArray *x, NDArray *z, std::vector<int> axis) : OpBenchmark(testName ,x, z, axis) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            PointersManager manager(LaunchContext::defaultContext(), "reductionBM");
            if (_z->isScalar())
                NativeOpExecutioner::execReduceFloatScalar(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo());
            else {
                auto dims = reinterpret_cast<int *>(manager.replicatePointer(_axis.data(), _axis.size() * sizeof(int)));

                shape::TAD tad;
                tad.init(_x->shapeInfo(), _axis.data(), _axis.size());
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)));
                auto tadOffsets = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong)));

                NativeOpExecutioner::execReduceFloat(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo(), dims, _axis.size(), tadOnlyShapeInfo, tadOffsets);
            }

            manager.synchronize();
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
            return result;
        }

        std::string inplace() override {
            return "n/a";
        }

        ~ReductionBenchmark(){
            delete _x;
            delete _z;
        }

        std::string axis() override {
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
            return new ReductionBenchmark((reduce::FloatOps) _opNum, _testName, _x, _z, _axis);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H
