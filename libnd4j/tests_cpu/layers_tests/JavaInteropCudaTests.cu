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

#include "testlayers.h"
#include <helpers/PointersManager.h>
#include <array/ExtraArguments.h>
#include <ops/declarable/CustomOperations.h>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace nd4j;
using namespace nd4j::ops;

class JavaInteropCudaTests : public testing::Test {
public:

};

TEST_F(JavaInteropCudaTests, test_DeclarableOp_execution_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    auto y = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {3, 5});
    x.assign(1.f);
    e.assign(2.f);

    nd4j::ops::add op;
    Context context(1);

    context.setCudaContext(LaunchContext::defaultContext()->getCudaStream(), LaunchContext::defaultContext()->getReductionPointer(), LaunchContext::defaultContext()->getAllocationPointer());
    context.setInputArray(0, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo());
    context.setInputArray(1, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo());

    context.setOutputArray(0, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo());

    PointersManager pm(LaunchContext::defaultContext(), "test_DeclarableOp_execution_1");
    execCustomOp2(nullptr, op.getOpHash(), &context);

    pm.synchronize();

    ASSERT_EQ(e, x);
}

TEST_F(JavaInteropCudaTests, test_DeclarableOp_execution_2) {
    NDArray x('c', {3, 1, 2}, nd4j::DataType::FLOAT32);
    NDArray y('c', {2, 2}, nd4j::DataType::FLOAT32);
    NDArray z('c', {3, 2, 2}, nd4j::DataType::BOOL);
    NDArray e('c', {3, 2, 2}, nd4j::DataType::BOOL);

    x.assign(1.f);
    y.assign(2.f);
    e.assign(false);

    nd4j::ops::equals op;
    Context context(1);

    context.setCudaContext(LaunchContext::defaultContext()->getCudaStream(), LaunchContext::defaultContext()->getReductionPointer(), LaunchContext::defaultContext()->getAllocationPointer());
    context.setInputArray(0, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo());
    context.setInputArray(1, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo());

    context.setOutputArray(0, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

    PointersManager pm(LaunchContext::defaultContext(), "test_DeclarableOp_execution_2");
    execCustomOp2(nullptr, op.getOpHash(), &context);

    pm.synchronize();

    ASSERT_EQ(e, z);
}

