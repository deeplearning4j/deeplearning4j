/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <ops/ops.h>
#include <helpers/GradCheck.h>
#include <helpers/RandomLauncher.h>
#include <exceptions/cuda_exception.h>


using namespace sd;


class AtomicTests : public testing::Test {
public:
    AtomicTests() {
        //
    }
};

template <typename T>
static _CUDA_G void multiplyKernel(void *vbuffer, uint64_t length, void *vresult) {
    auto buffer = reinterpret_cast<T*>(vbuffer);
    auto result = reinterpret_cast<T*>(vresult);

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto e = tid; e < length; e += gridDim.x * blockDim.x) {
        auto rem = e % 4;
        auto i = (e - rem) / 4;

        sd::math::atomics::nd4j_atomicMul<T>(&result[i], buffer[e]);
    }
}

template <typename T>
static void multiplyLauncher(void *vbuffer, uint64_t length, void *vresult) {
    multiplyKernel<T><<<256, 256, 1024, *sd::LaunchContext::defaultContext()->getCudaStream()>>>(vbuffer, length, vresult);
    auto err = cudaStreamSynchronize(*sd::LaunchContext::defaultContext()->getCudaStream());
    if (err != 0)
        sd::cuda_exception::build("multiply failed", err);
}

template <typename T>
static _CUDA_G void sumKernel(void *vbuffer, uint64_t length, void *vresult) {
    auto buffer = reinterpret_cast<T*>(vbuffer);
    auto result = reinterpret_cast<T*>(vresult);

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto e = tid; e < length; e += gridDim.x * blockDim.x) {
        auto rem = e % 4;
        auto i = (e - rem) / 4;

        sd::math::atomics::nd4j_atomicAdd<T>(&result[i], buffer[e]);
    }
}

template <typename T>
static void sumLauncher(void *vbuffer, uint64_t length, void *vresult) {
    sumKernel<T><<<256, 256, 1024, *sd::LaunchContext::defaultContext()->getCudaStream()>>>(vbuffer, length, vresult);
    auto err = cudaStreamSynchronize(*sd::LaunchContext::defaultContext()->getCudaStream());
    if (err != 0)
        sd::cuda_exception::build("sum failed", err);
}

template <typename T>
static _CUDA_G void subKernel(void *vbuffer, uint64_t length, void *vresult) {
    auto buffer = reinterpret_cast<T*>(vbuffer);
    auto result = reinterpret_cast<T*>(vresult);

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto e = tid; e < length; e += gridDim.x * blockDim.x) {
        auto rem = e % 4;
        auto i = (e - rem) / 4;

        sd::math::atomics::nd4j_atomicSub<T>(&result[i], buffer[e]);
    }
}

template <typename T>
static void subLauncher(void *vbuffer, uint64_t length, void *vresult) {
    subKernel<T><<<256, 256, 1024, *sd::LaunchContext::defaultContext()->getCudaStream()>>>(vbuffer, length, vresult);
    auto err = cudaStreamSynchronize(*sd::LaunchContext::defaultContext()->getCudaStream());
    if (err != 0)
        sd::cuda_exception::build("sub failed", err);
}

template <typename T>
static _CUDA_G void divKernel(void *vbuffer, uint64_t length, void *vresult) {
    auto buffer = reinterpret_cast<T*>(vbuffer);
    auto result = reinterpret_cast<T*>(vresult);

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto e = tid; e < length; e += gridDim.x * blockDim.x) {
        auto rem = e % 4;
        auto i = (e - rem) / 4;

        sd::math::atomics::nd4j_atomicDiv<T>(&result[i], buffer[e]);
    }
}

template <typename T>
static void divLauncher(void *vbuffer, uint64_t length, void *vresult) {
    divKernel<T><<<256, 256, 1024, *sd::LaunchContext::defaultContext()->getCudaStream()>>>(vbuffer, length, vresult);
    auto err = cudaStreamSynchronize(*sd::LaunchContext::defaultContext()->getCudaStream());
    if (err != 0)
        sd::cuda_exception::build("div failed", err);
}

static void multiplyHost(NDArray &input, NDArray &output) {
    BUILD_SINGLE_SELECTOR(input.dataType(), multiplyLauncher, (input.specialBuffer(), input.lengthOf(), output.specialBuffer()), NUMERIC_TYPES);
}

static void sumHost(NDArray &input, NDArray &output) {
    BUILD_SINGLE_SELECTOR(input.dataType(), sumLauncher, (input.specialBuffer(), input.lengthOf(), output.specialBuffer()), NUMERIC_TYPES);
}

static void subHost(NDArray &input, NDArray &output) {
    BUILD_SINGLE_SELECTOR(input.dataType(), subLauncher, (input.specialBuffer(), input.lengthOf(), output.specialBuffer()), FLOAT_TYPES);
}

static void divHost(NDArray &input, NDArray &output) {
    BUILD_SINGLE_SELECTOR(input.dataType(), divLauncher, (input.specialBuffer(), input.lengthOf(), output.specialBuffer()), FLOAT_TYPES);
}

TEST_F(AtomicTests, test_multiply) {
    std::vector<sd::DataType> dtypes = {sd::DataType::FLOAT32, sd::DataType::DOUBLE, sd::DataType::INT16, sd::DataType::HALF};

    for (auto t:dtypes) {
        nd4j_printf("Trying data type [%s]\n", DataTypeUtils::asString(t).c_str());
        NDArray input('c', {4, 25}, t);
        NDArray output('c', {input.lengthOf() / 4}, t);
        NDArray exp = output.ulike();

        input.assign(2);
        output.assign(2);
        exp.assign(32);

        multiplyHost(input, output);
        ASSERT_EQ(exp, output);
    }
}

TEST_F(AtomicTests, test_multiply_2) {
    std::vector<sd::DataType> dtypes = {sd::DataType::FLOAT32, sd::DataType::DOUBLE, sd::DataType::HALF, sd::DataType::BFLOAT16};

    for (auto t:dtypes) {
        nd4j_printf("Trying data type [%s]\n", DataTypeUtils::asString(t).c_str());
        NDArray input('c', {4, 25}, t);
        NDArray output('c', {input.lengthOf() / 4}, t);
        NDArray exp = output.ulike();

        input.assign(1.5);
        output.assign(2);
        exp.assign(10.125);

        multiplyHost(input, output);
//        output.printBuffer("multiply 2");
        ASSERT_EQ(exp, output);
    }
}

TEST_F(AtomicTests, test_sum) {
    std::vector<sd::DataType> dtypes = {sd::DataType::FLOAT32, sd::DataType::DOUBLE, sd::DataType::BFLOAT16, sd::DataType::HALF, sd::DataType::INT16};

    for (auto t:dtypes) {
        nd4j_printf("Trying data type [%s]\n", DataTypeUtils::asString(t).c_str());
        NDArray input('c', {4, 25}, t);
        NDArray output('c', {input.lengthOf() / 4}, t);
        NDArray exp = output.ulike();

        input.assign(1);
        output.assign(1);
        exp.assign(5);

        sumHost(input, output);
//        output.printIndexedBuffer("Sum");
        ASSERT_EQ(exp, output);
    }
}

TEST_F(AtomicTests, test_sub) {
    std::vector<sd::DataType> dtypes = {sd::DataType::FLOAT32, sd::DataType::DOUBLE, sd::DataType::HALF};

    for (auto t:dtypes) {
        nd4j_printf("Trying data type [%s]\n", DataTypeUtils::asString(t).c_str());
        NDArray input('c', {4, 25}, t);
        NDArray output('c', {input.lengthOf() / 4}, t);
        NDArray exp = output.ulike();

        input.assign(1);
        output.assign(5);
        exp.assign(1);

        subHost(input, output);
//        output.printBuffer("Sub");

        ASSERT_EQ(exp, output);
    }
}

TEST_F(AtomicTests, test_div) {
    std::vector<sd::DataType> dtypes = {sd::DataType::FLOAT32, sd::DataType::DOUBLE, sd::DataType::BFLOAT16, sd::DataType::HALF};

    for (auto t:dtypes) {
        nd4j_printf("Trying data type [%s]\n", DataTypeUtils::asString(t).c_str());
        NDArray input('c', {4, 25}, t);
        NDArray output('c', {input.lengthOf() / 4}, t);
        NDArray exp = output.ulike();

        input.assign(2);
        output.assign(32);
        exp.assign(2);

        divHost(input, output);
//        output.printBuffer("Div");
        ASSERT_EQ(exp, output);
    }
}