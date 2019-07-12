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

    std::string BenchmarkHelper::printHeader() {
        return std::string("TestName\tOpNum\tWarmup\tNumIter\tDataType\tInplace\tShape\tStrides\tAxis\tOrders\tavg (us)\tmedian (us)\tmin (us)\tmax (us)\tstdev (us)\n");
    }

    std::string BenchmarkHelper::benchmarkOperation(OpBenchmark &benchmark) {

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

        auto n = NDArrayFactory::create(timings, LaunchContext::defaultContext());

        auto stdev = n.varianceNumber(nd4j::variance::SummaryStatsStandardDeviation, false).e<double>(0);
        auto min = n.reduceNumber(nd4j::reduce::Min).e<Nd4jLong>(0);
        auto max = n.reduceNumber(nd4j::reduce::Max).e<Nd4jLong>(0);

        // opNum, DataType, Shape, average time, median time
        auto t = benchmark.dataType();
        auto s = benchmark.shape();
        auto strides = benchmark.strides();
        auto o = benchmark.orders();
        auto a = benchmark.axis();
        auto inpl = benchmark.inplace();

        std::string temp;
        temp.resize(65536);

        // printing out stuff
        snprintf(const_cast<char *>(temp.data()), temp.length(), "%s\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%.2f\n", benchmark.testName().c_str(), benchmark.opNum(),
                    _wIterations, _rIterations, t.c_str(), inpl.c_str(), s.c_str(), strides.c_str(), a.c_str(), o.c_str(),
                    nd4j::math::nd4j_floor<double, Nd4jLong>(sumT), median, min, max, stdev);

        auto pos = temp.find('\n');
        return temp.substr(0, pos + 1);
    }

    void BenchmarkHelper::benchmarkScalarOperation(scalar::Ops op, std::string testName, double value, NDArray &x, NDArray &z) {
        auto y = NDArrayFactory::create(x.dataType(), value);

        //for (uint i = 0; i < _wIterations; i++)
            //NativeOpExecutioner::execScalar(op, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), y.buffer(), y.shapeInfo(), nullptr);


        std::vector<Nd4jLong> timings(_rIterations);
        double sumT = 0.0;

        for (uint i = 0; i < _rIterations; i++) {
            auto timeStart = std::chrono::system_clock::now();

            //NativeOpExecutioner::execScalar(op, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), y.buffer(), y.shapeInfo(), nullptr);

            auto timeEnd = std::chrono::system_clock::now();
            auto loopTime = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart)).count();
            timings[i] = loopTime;
            sumT += loopTime;
        }
        sumT /= _rIterations;

        std::sort(timings.begin(), timings.end());
        Nd4jLong median = timings[_rIterations / 2];

        NDArray n = NDArrayFactory::create(timings, nullptr);
        double stdev = n.varianceNumber(nd4j::variance::SummaryStatsStandardDeviation, false).e<double>(0);
        Nd4jLong min = n.reduceNumber(nd4j::reduce::Min).e<Nd4jLong>(0);
        Nd4jLong max = n.reduceNumber(nd4j::reduce::Max).e<Nd4jLong>(0);

        // opNum, DataType, Shape, average time, median time
        auto t = DataTypeUtils::asString(x.dataType());
        auto s = ShapeUtils::shapeAsString(&x);
        auto stride = ShapeUtils::strideAsString(&x);
        stride += "/";
        stride += ShapeUtils::strideAsString(&z);
        std::string o;
        o += x.ordering();
        o += "/";
        o += z.ordering();
        std::string inpl;
        inpl += (x == z ? "true" : "false");

        // printing out stuff
        nd4j_printf("%s\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\tn/a\t%lld\t%lld\t%lld\t%lld\t%.2f\n", testName.c_str(), op,
                    _wIterations, _rIterations, t.c_str(), inpl.c_str(), s.c_str(), stride.c_str(), o.c_str(),
                    nd4j::math::nd4j_floor<double, Nd4jLong>(sumT), median, min, max, stdev);
    }

    std::string BenchmarkHelper::runOperationSuit(std::initializer_list<OpBenchmark*> benchmarks, const char *msg) {
        std::vector<OpBenchmark*> ops(benchmarks);
        return runOperationSuit(ops, msg);
    }

    std::string BenchmarkHelper::runOperationSuit(OpBenchmark* benchmark) {
        return benchmarkOperation(*benchmark);
    }

    std::string BenchmarkHelper::runOperationSuit(std::vector<OpBenchmark*> &benchmarks, bool postHeaders, const char *msg) {
        std::string result;

        if (msg != nullptr && postHeaders) {
            result += "\n";
            result += msg;
            result += "\n";
        }

        if (postHeaders)
            result += printHeader();

        for (auto v:benchmarks)
            result += benchmarkOperation(*v);

        return result;
    }

    std::string BenchmarkHelper::runOperationSuit(DeclarableBenchmark *op, const std::function<Context* (Parameters &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string result;

        if (message != nullptr) {
            result += "\n";
            result += message;
            result += "\n";
        }

        result += printHeader();

        std::vector<OpBenchmark*> list;

        for (auto &p : parameters) {
            auto ctx = func(p);

            auto clone = reinterpret_cast<DeclarableBenchmark*>(op->clone());
            clone->setContext(ctx);

            result += runOperationSuit(clone);

            delete clone;
        }

        return result;
    }

    std::string BenchmarkHelper::runOperationSuit(ScalarBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, z);
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

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<ScalarBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(ScalarBenchmark *op, const std::function<void (ResultSet&, ResultSet&)>& func, const char *message) {
        std::string output;

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

        output += runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<ScalarBenchmark*>(v);
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(TransformBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, z);
            std::vector<OpBenchmark *> result;

            if (x.size() != z.size())
                throw std::runtime_error("TransformBenchmark: number of X and Z arrays should match");

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setZ(z_);

                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<TransformBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(TransformBenchmark *op, const std::function<void (ResultSet&, ResultSet&)>& func, const char *message) {
        std::string output;

        ResultSet x;
        x.setNonRemovable();
        ResultSet z;
        z.setNonRemovable();
        func(x, z);
        std::vector<OpBenchmark*> result;

        if (x.size() != z.size())
            throw std::runtime_error("TransformBenchmark: number of X and Z arrays should match");

        for (int e = 0; e < x.size(); e++) {
            auto x_ = x.at(e);
            auto z_ = z.at(e);

            auto clone = op->clone();
            clone->setX(x_);
            clone->setZ(z_);

            result.emplace_back(clone);
        }

        output += runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<TransformBenchmark*>(v);
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(ReductionBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&)>& func, ParametersBatch &parametersBatch, const char *message) {
        std::string output;
        auto parameters = parametersBatch.parameters();

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, z);
            std::vector<OpBenchmark*> result;

            if (x.size() != z.size())
                throw std::runtime_error("ReductionBenchmark: number of X and Z arrays should match");

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setZ(z_);

                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<ReductionBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(ReductionBenchmark *op, const std::function<void (ResultSet&, ResultSet&)>& func, const char *message) {
        std::string output;
        ResultSet x;
        x.setNonRemovable();
        ResultSet z;
        z.setNonRemovable();
        func(x, z);
        std::vector<OpBenchmark*> result;

        if (x.size() != z.size())
            throw std::runtime_error("ReductionBenchmark: number of X and Z arrays should match");

        for (int e = 0; e < x.size(); e++) {
            auto x_ = x.at(e);
            auto z_ = z.at(e);

            auto clone = op->clone();
            clone->setX(x_);
            clone->setZ(z_);

            result.emplace_back(clone);
        }

        output += runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<ReductionBenchmark*>(v);
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(ReductionBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet y;
            y.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, y, z);
            std::vector<OpBenchmark*> result;

            if (x.size() != z.size() || x.size() != y.size())
                throw std::runtime_error("ReductionBenchmark: number of X and Z arrays should match");

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto y_ = y.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setZ(z_);

                if (y_ != nullptr) {
                    clone->setAxis(y_->asVectorT<int>());
                    delete y_;
                }

                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<ReductionBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(ReductionBenchmark *op, const std::function<void (ResultSet&, ResultSet&, ResultSet &)>& func, const char *message) {
        std::string output;

        ResultSet x;
        x.setNonRemovable();
        ResultSet y;
        y.setNonRemovable();
        ResultSet z;
        z.setNonRemovable();
        func(x, y, z);
        std::vector<OpBenchmark*> result;

        if (x.size() != z.size() || x.size() != y.size())
            throw std::runtime_error("ReductionBenchmark: number of X and Z arrays should match");

        for (int e = 0; e < x.size(); e++) {
            auto x_ = x.at(e);
            auto y_ = y.at(e);
            auto z_ = z.at(e);

            auto clone = op->clone();
            clone->setX(x_);
            clone->setZ(z_);

            if (y_ != nullptr) {
                clone->setAxis(y_->asVectorT<int>());
                delete y_;
            }
            result.emplace_back(clone);
        }

        output += runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<ReductionBenchmark*>(v);
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(BroadcastBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet y;
            y.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, y, z);
            std::vector<OpBenchmark*> result;

            if (x.size() != z.size() )
                throw std::runtime_error("BroadcastBenchmark: number of X and Z arrays should match");

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto y_ = y.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setY(y_);
                clone->setZ(z_);

                clone->setAxis(op->getAxis());
                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<BroadcastBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(PairwiseBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet y;
            y.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, y, z);
            std::vector<OpBenchmark*> result;

            if (x.size() != z.size() || x.size() != y.size())
                throw std::runtime_error("PairwiseBenchmark: number of X and Z arrays should match");

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto y_ = y.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setY(y_);
                clone->setZ(z_);

                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<PairwiseBenchmark*>(v);
            }
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(PairwiseBenchmark *op, const std::function<void (ResultSet&, ResultSet&, ResultSet &)>& func, const char *message) {
        std::string output;

        ResultSet x;
        x.setNonRemovable();
        ResultSet y;
        y.setNonRemovable();
        ResultSet z;
        z.setNonRemovable();
        func(x, y, z);
        std::vector<OpBenchmark*> result;

        if (x.size() != z.size() || x.size() != y.size())
            throw std::runtime_error("PairwiseBenchmark: number of X and Z arrays should match");

        for (int e = 0; e < x.size(); e++) {
            auto x_ = x.at(e);
            auto y_ = y.at(e);
            auto z_ = z.at(e);

            auto clone = op->clone();
            clone->setX(x_);
            clone->setY(y_);
            clone->setZ(z_);

            result.emplace_back(clone);
        }

        output += runOperationSuit(result, message);

        // removing everything
        for (auto v:result) {
            delete reinterpret_cast<PairwiseBenchmark*>(v);
        }

        return output;
    }

    std::string BenchmarkHelper::runOperationSuit(MatrixBenchmark *op, const std::function<void (Parameters &, ResultSet&, ResultSet&, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message) {
        auto parameters = parametersBatch.parameters();
        std::string output;

        if (message != nullptr) {
            output += "\n";
            output += message;
            output += "\n";
        }

        output += printHeader();

        for (auto &p: parameters) {
            ResultSet x;
            x.setNonRemovable();
            ResultSet y;
            y.setNonRemovable();
            ResultSet z;
            z.setNonRemovable();
            func(p, x, y, z);
            std::vector<OpBenchmark*> result;

            for (int e = 0; e < x.size(); e++) {
                auto x_ = x.at(e);
                auto y_ = y.at(e);
                auto z_ = z.at(e);

                auto clone = op->clone();
                clone->setX(x_);
                clone->setY(y_);
                clone->setZ(z_);

                result.emplace_back(clone);
            }

            output += runOperationSuit(result, false);

            // removing everything
            for (auto v:result) {
                delete reinterpret_cast<MatrixBenchmark*>(v);
            }
        }

        return output;
    }
}