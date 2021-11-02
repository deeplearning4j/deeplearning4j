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

/**
 * @author raver119@gmail.com
 */

#ifndef LIBND4J_BENCHMARKHELPER_H
#define LIBND4J_BENCHMARKHELPER_H

#include <array/NDArray.h>
#include <array/ResultSet.h>
#include <graph/Context.h>
#include <helpers/OpBenchmark.h>
#include <helpers/benchmark/BoolParameters.h>
#include <helpers/benchmark/BroadcastBenchmark.h>
#include <helpers/benchmark/DeclarableBenchmark.h>
#include <helpers/benchmark/IntParameters.h>
#include <helpers/benchmark/IntPowerParameters.h>
#include <helpers/benchmark/MatrixBenchmark.h>
#include <helpers/benchmark/PairwiseBenchmark.h>
#include <helpers/benchmark/Parameters.h>
#include <helpers/benchmark/ParametersBatch.h>
#include <helpers/benchmark/ParametersSpace.h>
#include <helpers/benchmark/PredefinedParameters.h>
#include <helpers/benchmark/ReductionBenchmark.h>
#include <helpers/benchmark/ScalarBenchmark.h>
#include <helpers/benchmark/TransformBenchmark.h>
#include <ops/declarable/DeclarableOp.h>

namespace sd {

class SD_LIB_EXPORT BenchmarkHelper {
 private:
  unsigned int _wIterations;
  unsigned int _rIterations;

 protected:
  std::string benchmarkOperation(OpBenchmark &benchmark);

  void benchmarkScalarOperation(scalar::Ops op, std::string testName, double value, NDArray &x, NDArray &z);

  void benchmarkDeclarableOp(sd::ops::DeclarableOp &op, std::string testName, Context &context);

  void benchmarkGEMM(char orderA, std::initializer_list<sd::LongType> shapeA, char orderB,
                     std::initializer_list<sd::LongType> shapeB, char orderC,
                     std::initializer_list<sd::LongType> shapeC);

  std::string printHeader();

 public:
  BenchmarkHelper(unsigned int warmUpIterations = 10, unsigned int runIterations = 100);

  std::string runOperationSuit(std::initializer_list<OpBenchmark *> benchmarks, const char *msg = nullptr);
  std::string runOperationSuit(std::vector<OpBenchmark *> &benchmarks, bool postHeaders, const char *msg = nullptr);
  std::string runOperationSuit(OpBenchmark *benchmark);

  std::string runOperationSuit(ScalarBenchmark *op, const std::function<void(ResultSet &, ResultSet &)> &func,
                               const char *message = nullptr);
  std::string runOperationSuit(TransformBenchmark *op, const std::function<void(ResultSet &, ResultSet &)> &func,
                               const char *message = nullptr);
  std::string runOperationSuit(ReductionBenchmark *op, const std::function<void(ResultSet &, ResultSet &)> &func,
                               const char *message = nullptr);
  std::string runOperationSuit(ReductionBenchmark *op,
                               const std::function<void(ResultSet &, ResultSet &, ResultSet &)> &func,
                               const char *message = nullptr);
  std::string runOperationSuit(PairwiseBenchmark *op,
                               const std::function<void(ResultSet &, ResultSet &, ResultSet &)> &func,
                               const char *message = nullptr);

  std::string runOperationSuit(TransformBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(ScalarBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(ReductionBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(ReductionBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(BroadcastBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(PairwiseBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
  std::string runOperationSuit(MatrixBenchmark *op,
                               const std::function<void(Parameters &, ResultSet &, ResultSet &, ResultSet &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);

  std::string runOperationSuit(DeclarableBenchmark *op, const std::function<Context *(Parameters &)> &func,
                               ParametersBatch &parametersBatch, const char *message = nullptr);
};
}  // namespace sd

#endif  // DEV_TESTS_BENCHMARKHELPER_H
