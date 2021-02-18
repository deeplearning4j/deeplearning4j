/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

using namespace sd::graph;

namespace sd {
    class ND4J_EXPORT ReductionBenchmark : public OpBenchmark {
    protected:
        int _opType;        //0=Float, 1=Same
    public:
        ReductionBenchmark() : OpBenchmark() {
            //
        }

        ReductionBenchmark(reduce::FloatOps op, std::string testName, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(testName, x, z, axis) {
            _opNum = (int) op;
            _opType = 0;
        }

        ReductionBenchmark(reduce::SameOps op, std::string testName, NDArray *x, NDArray *z, std::initializer_list<int> axis) : OpBenchmark(testName, x, z, axis) {
            _opNum = (int) op;
            _opType = 1;
        }


        ReductionBenchmark(reduce::FloatOps op) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 0;
        }

        ReductionBenchmark(reduce::FloatOps op, std::string testName) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 0;
            _testName = testName;
        }

        ReductionBenchmark(reduce::SameOps op) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 1;
        }

        ReductionBenchmark(reduce::SameOps op, std::string testName) : OpBenchmark() {
            _opNum = (int) op;
            _opType = 1;
            _testName = testName;
        }

        ReductionBenchmark(reduce::FloatOps op, std::string testName, NDArray *x, NDArray *z, std::vector<int> axis) : OpBenchmark(testName ,x, z, axis) {
            _opNum = (int) op;
            _opType = 0;
        }

        ReductionBenchmark(reduce::SameOps op, std::string testName, NDArray *x, NDArray *z, std::vector<int> axis) : OpBenchmark(testName ,x, z, axis) {
            _opNum = (int) op;
            _opType = 1;
        }

        void executeOnce() override {
            PointersManager manager(LaunchContext::defaultContext(), "reductionBM");

            if (_z->isScalar() || _y == nullptr)
                if (_opType == 0)
                    NativeOpExecutioner::execReduceFloatScalar(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo());
                else
                    NativeOpExecutioner::execReduceSameScalar(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo());
            else {
                auto pack = ConstantTadHelper::getInstance().tadForDimensions(_x->shapeInfo(), _axis);

                auto tadOnlyShapeInfo = Environment::getInstance().isCPU() ? pack.primaryShapeInfo() : pack.specialShapeInfo();
                auto tadOffsets = Environment::getInstance().isCPU() ? pack.primaryOffsets() : pack.specialOffsets();

                if (_opType == 0)
                    NativeOpExecutioner::execReduceFloat(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo(), nullptr, _axis.size());
                else
                    NativeOpExecutioner::execReduceSame(LaunchContext::defaultContext(), _opNum, _x->buffer(), _x->shapeInfo(), _x->specialBuffer(), _x->specialShapeInfo(), nullptr, _z->buffer(), _z->shapeInfo(), _z->specialBuffer(), _z->specialShapeInfo(), nullptr, _axis.size());
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
            if (_opType == 0)
                return new ReductionBenchmark((reduce::FloatOps) _opNum, _testName, _x, _z, _axis);
            else
                return new ReductionBenchmark((reduce::SameOps) _opNum, _testName, _x, _z, _axis);
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H