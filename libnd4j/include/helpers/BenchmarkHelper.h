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

#ifndef LIBND4J_BENCHMARKHELPER_H
#define LIBND4J_BENCHMARKHELPER_H


#include <helpers/OpBenchmark.h>
#include <helpers/benchmark/ScalarBenchmark.h>
#include <helpers/benchmark/TransformBenchmark.h>
#include <helpers/benchmark/ReductionBenchmark.h>
#include <helpers/benchmark/PairwiseBenchmark.h>
#include <helpers/benchmark/DeclarableBenchmark.h>
#include <helpers/benchmark/MatrixBenchmark.h>
#include <helpers/benchmark/BroadcastBenchmark.h>
#include <ops/declarable/DeclarableOp.h>
#include <graph/Context.h>
#include <NDArray.h>
#include <benchmark/Parameters.h>
#include <benchmark/PredefinedParameters.h>
#include <benchmark/ParametersBatch.h>
#include <benchmark/ParametersSpace.h>
#include <benchmark/BoolParameters.h>
#include <benchmark/IntParameters.h>
#include <benchmark/IntPowerParameters.h>
#include <array/ResultSet.h>

namespace nd4j {

    class BenchmarkHelper {
    private:
        unsigned int _wIterations;
        unsigned int _rIterations;

    protected:
        void benchmarkOperation(OpBenchmark &benchmark);

        void benchmarkScalarOperation(scalar::Ops op, std::string testName, double value, NDArray &x, NDArray &z);

        void benchmarkDeclarableOp(nd4j::ops::DeclarableOp &op, std::string testName, Context &context);

        void benchmarkGEMM(char orderA, std::initializer_list<Nd4jLong> shapeA, char orderB, std::initializer_list<Nd4jLong> shapeB, char orderC, std::initializer_list<Nd4jLong> shapeC);

        void printHeader();
    public:
        BenchmarkHelper(unsigned int warmUpIterations = 10, unsigned int runIterations = 100);

        void runOperationSuit(std::initializer_list<OpBenchmark*> benchmarks, const char *msg = nullptr);
        void runOperationSuit(std::vector<OpBenchmark*> &benchmarks, bool postHeaders, const char *msg = nullptr);

        void runOperationSuit(ScalarBenchmark *op, const std::function<void (ResultSet &, ResultSet &)>& func, const char *message = nullptr);
        void runOperationSuit(TransformBenchmark *op, const std::function<void (ResultSet &, ResultSet &)>& func, const char *message = nullptr);
        void runOperationSuit(ReductionBenchmark *op, const std::function<void (ResultSet &, ResultSet &)>& func, const char *message = nullptr);
        void runOperationSuit(ReductionBenchmark *op, const std::function<void (ResultSet &, ResultSet &, ResultSet &)>& func, const char *message = nullptr);
        void runOperationSuit(PairwiseBenchmark *op, const std::function<void (ResultSet &, ResultSet &, ResultSet &)>& func, const char *message = nullptr);


        void runOperationSuit(TransformBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(ScalarBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(ReductionBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(ReductionBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(BroadcastBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(PairwiseBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);
        void runOperationSuit(MatrixBenchmark *op, const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);

        void runOperationSuit(DeclarableBenchmark *op, const std::function<Context* (Parameters &)>& func, ParametersBatch &parametersBatch, const char *message = nullptr);


        void runScalarSuit();

        void runAllSuits();
    };
}


#endif //DEV_TESTS_BENCHMARKHELPER_H
