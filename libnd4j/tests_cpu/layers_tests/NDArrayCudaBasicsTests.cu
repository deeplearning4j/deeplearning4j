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
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/LaunchContext.h>
#include <specials_cuda.h>
#include <TAD.h>

#include <cuda.h>
#include <cuda_launch_config.h>

using namespace nd4j;
using namespace nd4j::graph;

class NDArrayCudaBasicsTests : public testing::Test {
public:

};

TEST_F(NDArrayCudaBasicsTests, Test_Registration_1) {
    auto x = NDArrayFactory::create<int>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<int>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_3) {
    auto x = NDArrayFactory::create<int>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<int>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());

    NDArray::registerSpecialUse({&x}, {&y});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());

    ASSERT_TRUE(y.isActualOnDeviceSide());
    ASSERT_TRUE(y.isActualOnHostSide());
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_01) {
    auto x = NDArrayFactory::create_<int>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create_<int>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());
    delete x;
    delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_02) {
    auto x = NDArrayFactory::create_<int>('c', {5});
    auto y = NDArrayFactory::create_<int>('c', {5});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_FALSE(x->isActualOnHostSide());
    delete x;
    delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_03) {
    auto x = NDArrayFactory::create_<int>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create_<int>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());

    NDArray::registerSpecialUse({y}, {x});
    x->applyTransform(transform::Neg, y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //y->syncToHost();
    y->printBuffer("Negatives");
    delete x;
    delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_Cosine_1) {
    auto x = NDArrayFactory::create_<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create_<double>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());

    NDArray::registerSpecialUse({y}, {x});
    x->applyTransform(transform::Cosine, y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //y->syncToHost();
    y->printBuffer("Cosine");
    delete x;
    delete y;
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_1) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto z = NDArrayFactory::create<double>('c', { 5 }, {10, 10, 10, 10, 10});

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);

    Nd4jPointer nativeStream = (Nd4jPointer)malloc(sizeof(cudaStream_t));
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream");
    cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
    auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    //cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
    //cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);

    LaunchContext lc(stream, nullptr, nullptr);
    NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr);
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    //double* localBuffer = ;
    cudaMemcpy(z.buffer(), z.specialBuffer(), z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost);
    res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    x.printBuffer("X = ");
    y.printBuffer("Y = ");
    z.printBuffer("Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_2) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);

    Nd4jPointer nativeStream = (Nd4jPointer)malloc(sizeof(cudaStream_t));
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream");
    cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
    auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    //cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
    //cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);

    LaunchContext lc(stream, *stream, nullptr, nullptr);
    NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr);
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);

    cudaMemcpyAsync(z.buffer(), z.specialBuffer(), z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost, *stream);
    res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    z.printBuffer("2Result out");
    //cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_3) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto z = NDArrayFactory::create<double>('c', { 5 }, {10, 10, 10, 10, 10});

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);

    Nd4jPointer nativeStream = (Nd4jPointer)malloc(sizeof(cudaStream_t));
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream");
    cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
    auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    //cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
    //cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);

    LaunchContext lc(stream, *stream, nullptr, nullptr);
    NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr);
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    //double* localBuffer = ;
    cudaMemcpy(z.buffer(), z.specialBuffer(), z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost);
    res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    x.printBuffer("3X = ");
    y.printBuffer("3Y = ");
    z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_4) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x.applyPairwiseTransform(pairwise::Add, &y, &z, nullptr);
    z.syncToHost();
    x.printBuffer("3X = ");
    y.printBuffer("3Y = ");
    z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_5) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x += y;
    //x.applyPairwiseTransform(pairwise::Add, &y, &z, nullptr);
    x.syncToHost();
    x.printBuffer("33X = ");
    //y.printBuffer("3Y = ");
    //z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < x.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_6) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>(2); //.'c', { 5 }, { 1, 2, 3, 4, 5});
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 3, 4, 5, 6, 7 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x += y;
    //x.applyPairwiseTransform(pairwise::Add, &y, &z, nullptr);
    x.syncToHost();
    x.printBuffer("6X = ");
    //y.printBuffer("3Y = ");
    //z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < x.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_7) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    //auto y = NDArrayFactory::create<double>(2); //.'c', { 5 }, { 1, 2, 3, 4, 5});
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 3, 4, 5, 6, 7 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x += 2.;
    //x.applyPairwiseTransform(pairwise::Add, &y, &z, nullptr);
    x.syncToHost();
    x.printBuffer("7X = ");
    //y.printBuffer("3Y = ");
    //z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < x.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestMultiply_1) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 1, 4, 9, 16, 25 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    z.syncToHost();
    x.printBuffer("3X = ");
    y.printBuffer("3Y = ");
    z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestMultiply_2) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    NDArray z('c', { 5 }, nd4j::DataType::DOUBLE);

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 1, 4, 9, 16, 25 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    x.printBuffer("3X = ");
    y.printBuffer("3Y = ");
    z.syncToHost();
    z.printBuffer("3Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestMultiply_3) {
    // allocating host-side arrays
    NDArray x('c', { 5 }, { 1, 2, 3, 4, 5}, nd4j::DataType::DOUBLE);
    NDArray y('c', { 5 }, { 1., 2., 3., 4., 5.}, nd4j::DataType::DOUBLE);
    auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 1, 4, 9, 16, 25 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    //x.printBuffer("23X = ");
    //y.printBuffer("23Y = ");
    z.syncToHost();
    z.printBuffer("23Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestMultiply_4) {
    // allocating host-side arrays
    NDArray x('c', { 5 }, { 1, 2, 3, 4, 5}, nd4j::DataType::DOUBLE);
    NDArray y('c', { 5 }, { 1., 2., 3., 4., 5.}, nd4j::DataType::DOUBLE);
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 1, 4, 9, 16, 25 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    //x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    //x.printBuffer("23X = ");
    //y.printBuffer("23Y = ");
    x *= y;
    x.syncToHost();
    x.printBuffer("33Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    for (int e = 0; e < x.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    }
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestPrimitiveNeg_01) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<int>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<int>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto exp = NDArrayFactory::create<int>('c', { 5 }, { -1, -2, -3, -4, -5 });

    auto stream = x.getContext()->getCudaStream();//reinterpret_cast<cudaStream_t *>(&nativeStream);

    NativeOpExecutioner::execTransformSame(x.getContext(), transform::Neg, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), nullptr, nullptr, nullptr);
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    y.syncToHost();

    x.printBuffer("X = ");
    y.printBuffer("Y = ");

    for (int e = 0; e < y.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<int>(e), y.e<int>(e), 1e-5);
    }
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveNeg_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());

    x.applyTransform(transform::Neg, &y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    y.syncToHost();
    y.printBuffer("Negatives2");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveCosine_1) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());

    x.applyTransform(transform::Cosine, &y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    y.syncToHost();
    y.printBuffer("Cosine2");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveCosine_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    auto exp = NDArrayFactory::create<double>({0.540302, -0.416147, -0.989992, -0.653644, 0.283662});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());

    x.applyTransform(transform::Cosine, &y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    //y.syncToHost();
    y.printBuffer("PrimitiveCosine2");
    ASSERT_TRUE(exp.equalsTo(y));
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply) {
    // allocating host-side arrays
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, nd4j::DataType::DOUBLE);
    NDArray y('c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 2, 6, 12, 8, 15, 24 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    //x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    //x.printBuffer("23X = ");
    //y.printBuffer("23Y = ");
    x *= y;
    x.syncToHost();
    x.printBuffer("55Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    //for (int e = 0; e < x.lengthOf(); e++) {
    //    ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    //}
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_2) {
    // allocating host-side arrays
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, nd4j::DataType::DOUBLE);
    NDArray y('c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 11,12, 13,14, 15, 16 });
    auto expZ = NDArrayFactory::create<double>('c', { 2, 3 }, { 2, 6, 12, 8, 15, 24 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    //x.applyPairwiseTransform(pairwise::Multiply, &y, &z, nullptr);
    //x.printBuffer("23X = ");
    //y.printBuffer("23Y = ");
    //void NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, ExtraArguments *extraArgs)
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &y, &exp);
    exp.syncToHost();
    exp.printBuffer("55Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);

    //for (int e = 0; e < x.lengthOf(); e++) {
    //    ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
    //}
    ASSERT_TRUE(exp.equalsTo(expZ));

}


//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestReduceSum_1) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>(15);
    auto exp = NDArrayFactory::create<double>(15);

    auto stream = x.getContext()->getCudaStream();//reinterpret_cast<cudaStream_t *>(&nativeStream);

    NativeOpExecutioner::execReduceSameScalar(x.getContext(), reduce::Sum, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo());
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    y.syncToHost();

    x.printBuffer("X = ");
    y.printBuffer("Y = ");
    ASSERT_NEAR(y.e<double>(0), 15, 1e-5);
}
