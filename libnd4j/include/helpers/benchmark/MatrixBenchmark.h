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
// @author raver119@gmail.com
//

#include "../OpBenchmark.h"
#include <helpers/BlasHelper.h>
#include <MmulHelper.h>

#ifndef DEV_TESTS_MATRIXBENCHMARK_H
#define DEV_TESTS_MATRIXBENCHMARK_H

namespace nd4j {
    class ND4J_EXPORT MatrixBenchmark : public OpBenchmark {
    private:
        float _alpha = 1.0f;
        float _beta = 0.0f;
        bool _tA;
        bool _tB;
    public:
        MatrixBenchmark() : OpBenchmark() {
            //
        }

        MatrixBenchmark(float alpha, float beta, std::string testName, NDArray *x, NDArray *y, NDArray *z) : OpBenchmark(testName, x, y, z) {
            _alpha = alpha;
            _beta = beta;
            _tA = false;
            _tB = false;
        }

        MatrixBenchmark(float alpha, float beta, bool tA, bool tB, std::string name) : OpBenchmark() {
            _testName = name;
            _alpha = alpha;
            _beta = beta;
            _tA = tA;
            _tB = tB;
        }

        ~MatrixBenchmark(){
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
            auto xT = (_tA ? _x->transpose() : _x);
            auto yT = (_tB ? _y->transpose() : _y);

            MmulHelper::mmul(xT, yT, _z, _alpha, _beta);
        }

        std::string axis() override {
            return "N/A";
        }

        std::string inplace() override {
            return "N/A";
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

        std::string shape() override {
            std::string result;
            result += ShapeUtils::shapeAsString(_x);
            result += "x";
            result += ShapeUtils::shapeAsString(_y);
            result += "=";
            result += _z == nullptr ? "" : ShapeUtils::shapeAsString(_z);
            return result;
        }

        OpBenchmark* clone() override  {
            MatrixBenchmark* mb = new MatrixBenchmark(_alpha, _beta, _testName, _x, _y, _z);
            mb->_tA = _tA;
            mb->_tB = _tB;
            return mb;
        }
    };
}

#endif //DEV_TESTS_SCALARBENCHMARK_H