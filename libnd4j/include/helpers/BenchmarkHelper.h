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


#include <ops/declarable/DeclarableOp.h>
#include <NDArray.h>

namespace nd4j {
    class BenchmarkHelper {
    private:
        unsigned int _wIterations;
        unsigned int _rIterations;

    protected:
        void benchmarkScalarOperation(scalar::Ops op, double value, std::initializer_list<Nd4jLong> shape, nd4j::DataType dataTypeX);

        template <typename X>
        void benchmarkPairwiseOperation(pairwise::Ops op, std::initializer_list<Nd4jLong> shape, nd4j::DataType dataTypeX);

        template <typename X, typename Z>
        void benchmarkTransformFloatOperation(transform::FloatOps op, std::initializer_list<Nd4jLong> shape, nd4j::DataType dataTypeX, nd4j::DataType dataTypeZ);

        template <typename X>
        void benchmarkTransformStrictOperation(transform::FloatOps op, std::initializer_list<Nd4jLong> shape, nd4j::DataType dataTypeX);

        void benchmarkDeclarableOp(nd4j::ops::DeclarableOp &op, Context &context);

        void benchmarkGEMM(char orderA, std::initializer_list<Nd4jLong> shapeA, char orderB, std::initializer_list<Nd4jLong> shapeB, char orderC, std::initializer_list<Nd4jLong> shapeC);
    public:
        BenchmarkHelper(unsigned int warmUpIterations = 10, unsigned int runIterations = 100);

        void runScalarSuit();

        void runAllSuits();
    };
}


#endif //DEV_TESTS_BENCHMARKHELPER_H
