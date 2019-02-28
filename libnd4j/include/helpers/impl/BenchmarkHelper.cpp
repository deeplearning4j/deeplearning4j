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

/**
 * @author raver119@gmail.com
 */


#include "../BenchmarkHelper.h"
#include <NDArrayFactory.h>
#include <chrono>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    BenchmarkHelper::BenchmarkHelper(unsigned int warmUpIterations, unsigned int runIterations) {
        _wIterations = warmUpIterations;
        _rIterations = runIterations;
    }

    void BenchmarkHelper::benchmarkOperation(OpBenchmark &benchmark) {

        for (uint i = 0; i < _wIterations; i++)
            benchmark.executeOnce();

        std::vector<Nd4jLong> timings(_rIterations);
        double sumT = 0.0;

        for (uint i = 0; i < _rIterations; i++) {
            auto timeStart = std::chrono::system_clock::now();

            benchmark.executeOnce();

            auto timeEnd = std::chrono::system_clock::now();
            auto loopTime = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart)).count();
            timings[i] = loopTime;
            sumT += loopTime;
        }
        sumT /= _rIterations;

        std::sort(timings.begin(), timings.end());
        Nd4jLong median = timings[_rIterations / 2];

        // opNum, DataType, Shape, average time, median time
        auto t = DataTypeUtils::asString(benchmark.x().dataType());
        auto s = ShapeUtils::shapeAsString(&benchmark.x());
        auto o = benchmark.orders();

        // printing out stuff
        nd4j_printf("%i\t%s\t%s\t%s\t%lld\t%lld\n", benchmark.opNum(), t.c_str(), s.c_str(), o.c_str(), nd4j::math::nd4j_floor<double, Nd4jLong>(sumT), median);
    }

    void BenchmarkHelper::benchmarkScalarOperation(scalar::Ops op, double value, NDArray &x, NDArray &z) {
        auto y = NDArrayFactory::create(x.dataType(), value);

        for (uint i = 0; i < _wIterations; i++)
            NativeOpExcutioner::execScalar(op, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), y.buffer(), y.shapeInfo(), nullptr);


        std::vector<Nd4jLong> timings(_rIterations);
        double sumT = 0.0;

        for (uint i = 0; i < _rIterations; i++) {
            auto timeStart = std::chrono::system_clock::now();

            NativeOpExcutioner::execScalar(op, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), y.buffer(), y.shapeInfo(), nullptr);

            auto timeEnd = std::chrono::system_clock::now();
            auto loopTime = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart)).count();
            timings[i] = loopTime;
            sumT += loopTime;
        }
        sumT /= _rIterations;

        std::sort(timings.begin(), timings.end());
        Nd4jLong median = timings[_rIterations / 2];

        // opNum, DataType, Shape, average time, median time
        auto t = DataTypeUtils::asString(x.dataType());
        auto s = ShapeUtils::shapeAsString(&x);

        // printing out stuff
        nd4j_printf("%i\t%s\t%s\t%lld\t%lld\n", op, t.c_str(), s.c_str(), nd4j::math::nd4j_floor<double, Nd4jLong>(sumT), median);
    }

    void BenchmarkHelper::runOperationSuit(std::initializer_list<OpBenchmark*> benchmarks, const char *msg) {
        std::vector<OpBenchmark*> ops(benchmarks);
        runOperationSuit(ops, msg);
    }

    void BenchmarkHelper::runOperationSuit(std::vector<OpBenchmark*> &benchmarks, const char *msg) {
        if (msg != nullptr) {
            nd4j_printf("%s\n", msg);
        }

        nd4j_printf("OpNum\tDataType\tShape\tOrders\tavg (us)\tmedian (us)\n","");

        for (auto v:benchmarks)
            benchmarkOperation(*v);

        nd4j_printf("\n","");
    }

    void BenchmarkHelper::runScalarSuit() {
        nd4j_printf("OpNum\tDataType\tShape\tOrders\tavg (us)\tmedian (us)\n","");

        std::initializer_list<std::initializer_list<Nd4jLong>> shapes = {{100}, {32, 256}, {32, 150, 200}, {32, 3, 244, 244}, {32, 64, 128, 256}};
        std::initializer_list<nd4j::DataType> dataTypes = {nd4j::DataType::FLOAT32, nd4j::DataType::DOUBLE};
        std::initializer_list<nd4j::scalar::Ops> ops = {scalar::Add, scalar::Divide, scalar::Pow};

        for (const auto &d:dataTypes) {
            for (const auto &o:ops) {
                for (const auto &s:shapes) {
                    //benchmarkScalarOperation(o, 2.0, s, d);
                }
            }
        }
    }


    void BenchmarkHelper::runOperationSuit(ScalarBenchmark *op, const std::function<void (ResultSet&, ResultSet&)>& func, const char *message) {
        ResultSet x;
        x.setNonRemovable();
        ResultSet z;
        z.setNonRemovable();
        func(x, z);
        std::vector<OpBenchmark*> result;

        if (x.size() != z.size())
            throw std::runtime_error("ScalarBenchmark: number of X and Z arrays should match");

        for (int e = 0; e < x.size(); e++) {
            auto x_ = x.at(e);
            auto z_ = z.at(e);

            auto clone = op->clone();
            clone->setX(x_);
            clone->setZ(z_);

            result.emplace_back(clone);
        }

        runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<ScalarBenchmark*>(v);
        }
    }
}