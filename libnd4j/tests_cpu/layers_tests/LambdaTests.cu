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

#include "testlayers.h"
#include <array/ExtraArguments.h>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace sd;

class LambdaTests : public testing::Test {
public:

    LambdaTests() {
        printf("\n");
        fflush(stdout);
    }
};

template <typename Lambda>
__global__ void runLambda(double *input, double *output, Nd4jLong length, Lambda lambda) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (Nd4jLong e = tid; e < length; e += gridDim.x * blockDim.x) {
        output[e] = lambda(input[e]);
    }
}

void launcher(cudaStream_t *stream, double *input, double *output, Nd4jLong length) {
    //auto f = [] __host__ __device__ (double x) -> double {
    //        return x + 1.;
    //};
    auto f = LAMBDA_D(x) {
        return x+1.;
    };


    runLambda<<<128, 128, 128, *stream>>>(input, output, length, f);
}


TEST_F(LambdaTests, test_basic_1) {
    auto x = NDArrayFactory::create<double>('c', {5});
    auto e = NDArrayFactory::create<double>('c', {5}, {1., 1., 1., 1., 1.});



    //x.applyLambda<double>(f, nullptr);
    launcher(LaunchContext::defaultContext()->getCudaStream(), (double *)x.specialBuffer(), (double *)x.specialBuffer(), x.lengthOf());
    auto res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
    ASSERT_EQ(0, res);

    ASSERT_EQ(e, x);
}

void test(NDArray &x) {
    auto f = LAMBDA_D(x) {
        return x+1.;
    };

    x.applyLambda(f, x);
}

template <typename T>
void test2(NDArray &x) {
    auto f = LAMBDA_T(x) {
        return x+1.;
    };

    x.applyLambda(f, x);
}

void testPairwise(NDArray &x, NDArray &y) {
    auto f = LAMBDA_DD(x, y) {
        return x + y +1.;
    };

    x.applyPairwiseLambda(y, f, x);
}

void testTriplewise(NDArray &i, NDArray &j, NDArray &k) {
    auto f = LAMBDA_DDD(i, j, k) {
        return i + j + k + 2.;
    };

    i.applyTriplewiseLambda(j, k, f, i);
}

void testIndexed(NDArray &x) {
    auto f = ILAMBDA_D(x) {
        return _idx + 1.;
    };

    x.applyIndexedLambda(f, x);
}

void testIndexedPairwise(NDArray &x, NDArray &y) {
    auto f = ILAMBDA_DD(x, y) {
        return _idx + x + y +1.;
    };

    x.applyIndexedPairwiseLambda(y, f, x);
}

TEST_F(LambdaTests, test_basic_2) {
    auto x = NDArrayFactory::create<double>('c', {5});
    auto e = NDArrayFactory::create<double>('c', {5}, {1., 1., 1., 1., 1.});

    test(x);

    ASSERT_EQ(e, x);
}

TEST_F(LambdaTests, test_basic_3) {
    auto x = NDArrayFactory::create<float>('c', {5});
    auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});

    test(x);

    ASSERT_EQ(e, x);
}

TEST_F(LambdaTests, test_basic_4) {
    auto x = NDArrayFactory::create<float>('c', {5});
    auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});

    test2<float>(x);

    ASSERT_EQ(e, x);
}

TEST_F(LambdaTests, test_basic_5) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1., 1., 1., 1., 1.});
    auto y = NDArrayFactory::create<double>('c', {5}, {2., 2., 2., 2., 2.});
    auto e = NDArrayFactory::create<double>('c', {5}, {4., 4., 4., 4., 4.});

    testPairwise(x, y);

    ASSERT_EQ(e, x);
}

TEST_F(LambdaTests, test_basic_6) {
    auto x = NDArrayFactory::create<double>('c', {5});
    auto e = NDArrayFactory::create<double>('c', {5}, {1., 2., 3., 4., 5.});

    testIndexed(x);

    ASSERT_EQ(e, x);
}

TEST_F(LambdaTests, test_basic_7) {
    auto w = NDArrayFactory::create<double>('c', {5}, {0., 0., 0., 0., 0.});
    auto x = NDArrayFactory::create<double>('c', {5}, {1., 1., 1., 1., 1.});
    auto y = NDArrayFactory::create<double>('c', {5}, {2., 2., 2., 2., 2.});
    auto e = NDArrayFactory::create<double>('c', {5}, {5., 5., 5., 5., 5.});

    testTriplewise(w, x, y);

    ASSERT_EQ(e, w);
}

TEST_F(LambdaTests, test_basic_8) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1., 1., 1., 1., 1.});
    auto y = NDArrayFactory::create<double>('c', {5}, {2., 2., 2., 2., 2.});
    auto e = NDArrayFactory::create<double>('c', {5}, {4., 5., 6., 7., 8.});

    testIndexedPairwise(x, y);

    ASSERT_EQ(e, x);
}


template <typename T>
void testPairwiseMy(NDArray &x, NDArray &y, NDArray &z) {

    auto f = LAMBDA_TT(x, y){
        return sd::math::nd4j_max<T>(x, (T)0.f)
              - x * y
              + sd::math::nd4j_log<T,T>((T)1.f
                + sd::math::nd4j_exp<T,T>(-sd::math::nd4j_abs(x)));
    };

    x.applyPairwiseLambda(y, f, z);
}

///////////////////////////////////////////////////////////////////
TEST_F(LambdaTests, test_basic_9) {

    NDArray labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray logits('c', {2,3,4}, sd::DataType::DOUBLE);
    NDArray output('c', {2,3,4}, sd::DataType::DOUBLE);
    NDArray expected('c', {2,3,4}, {0.744397, 0.598139, 0.554355, 0.913015, 0.474077, 1.037488, 0.403186, 1.171101, 0.341154, 1.313262, 0.287335, 1.463282, 0.241008, 1.620417, 0.201413, 1.783901, 0.167786, 1.952978, 2.039387, 0.126928, 0.115520, 2.305083, 0.095545, 2.486836});

    logits.linspace(0.1, 0.1);

    NDArray::prepareSpecialUse({&output}, {&logits, &labels});
    testPairwiseMy<double>(logits, labels, output);
    NDArray::registerSpecialUse({&output}, {&logits, &labels});

    // output.printBuffer(nullptr, -1, true);
    ASSERT_TRUE(expected.equalsTo(output));
}
