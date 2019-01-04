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

//////////////////////////////////////////////////////////////////////////
static cudaError_t allocateDeviceMem(LaunchContext& lc, std::vector<void*>& devicePtrs, const std::vector<std::pair<void*,size_t>>& hostData) {

    if(devicePtrs.size() != hostData.size())
        throw std::invalid_argument("prepareDataForCuda: two input sts::vectors should same sizes !");

    cudaError_t cudaResult;

    void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024);			if(cudaResult != 0) return cudaResult;
    int* allocationPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024);			if(cudaResult != 0) return cudaResult;

    lc.setReductionPointer(reductionPointer);
    lc.setAllocationPointer(allocationPointer);
    cudaStream_t stream = *lc.getCudaStream();

    for(int i = 0; i < devicePtrs.size(); ++i) {

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&devicePtrs[i]), hostData[i].second); if(cudaResult != 0) return cudaResult;
        cudaMemcpyAsync(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice, stream);
    }
    return cudaResult;
}

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
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));
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
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));
    cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
    auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    //cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
    //cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);
    z.lazyAllocateBuffer();
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
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));
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
    y.lazyAllocateBuffer();
    x.applyTransform(transform::Cosine, &y, nullptr);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    y.syncToHost();
    //exp.syncToHost();
    y.printBuffer("PrimitiveCosine2");
    exp.printBuffer("Primitive Cosine exp");
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.dataType() == y.dataType());
    for (int e = 0; e < y.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<double>(e), y.e<double>(e), 1e-5);
    }

    ASSERT_TRUE(exp.equalsTo(y));
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, TestRawBroadcast_2) {

    //if (!Environment::getInstance()->isExperimentalBuild())
    //    return;

    NDArray x = NDArrayFactory::create<double>('c', {2,3,4});
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, nd4j::DataType::DOUBLE);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
//    NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,3,4}, {10., 40., 90., 160., 50., 120., 210., 320., 90., 200., 330., 480., 650., 840., 1050., 1280., 850., 1080., 1330., 1600., 1050., 1320., 1610., 1920.}, nd4j::DataType::DOUBLE);
    x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
    hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));							// 0 -- dimensions
    hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
    hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
    std::vector<void*> devicePtrs(hostData.size(), nullptr);

    // create cuda stream and LaunchContext
    cudaError_t cudaResult;
    cudaStream_t stream;
    cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
    LaunchContext lc(&stream);

    // allocate required amount of global device memory and copy host data to it
    cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

    // call cuda kernel which calculates result
    NativeOpExecutioner::execBroadcast(&lc, nd4j::broadcast::Multiply,
                                       nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.syncToHost();
    z.printBuffer("Result with Broadcast2 (multiply)");
    // verify results
    for (int e = 0; e < z.lengthOf(); e++)
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

    // free allocated global device memory
    for(int i = 0; i < devicePtrs.size(); ++i)
        cudaFree(devicePtrs[i]);

    // delete cuda stream
    cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

TEST_F(NDArrayCudaBasicsTests, TestRawBroadcast_3) {

    //if (!Environment::getInstance()->isExperimentalBuild())
    //    return;

    NDArray x('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, nd4j::DataType::DOUBLE);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
//    NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,3,4}, {10., 40., 90., 160., 50., 120., 210., 320., 90., 200., 330., 480., 650., 840., 1050., 1280., 850., 1080., 1330., 1600., 1050., 1320., 1610., 1920.}, nd4j::DataType::DOUBLE);
    x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
    hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));							// 0 -- dimensions
    hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
    hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
    std::vector<void*> devicePtrs(hostData.size(), nullptr);

    // create cuda stream and LaunchContext
    cudaError_t cudaResult;
    //cudaStream_t stream;
    //cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
    LaunchContext* pLc = x.getContext();//(&stream);
    cudaStream_t* stream = pLc->getCudaStream();
    // allocate required amount of global device memory and copy host data to it
    cudaResult = allocateDeviceMem(*pLc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
    NDArray::registerSpecialUse({&z}, {&x, &y});
    // call cuda kernel which calculates result
    NativeOpExecutioner::execBroadcast(pLc, nd4j::broadcast::Multiply,
                                       nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    //cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.syncToHost();
    z.printBuffer("Result with Broadcast3 (multiply)");
    // verify results
    for (int e = 0; e < z.lengthOf(); e++)
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

    // free allocated global device memory
    for(int i = 0; i < devicePtrs.size(); ++i)
        cudaFree(devicePtrs[i]);
    ASSERT_TRUE(exp.equalsTo(z));
    // delete cuda stream
    //cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}


TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_1) {
    // allocating host-side arrays
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(3.); //'c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    //auto z = NDArrayFactory::create<double>('c', { 5 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 3, 6, 9, 12, 15, 18 });

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
    x.printBuffer("54Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    ASSERT_TRUE(exp.equalsTo(x));
//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), x.e<double>(e), 1e-5);
//    }
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_01) {
    // allocating host-side arrays
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(3.); //'c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    auto z = NDArrayFactory::create<double>('c', { 2, 3 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 3, 6, 9, 12, 15, 18 });

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
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &y, &z);// *= y;
    z.syncToHost();
    z.printBuffer("53Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    ASSERT_TRUE(exp.equalsTo(z));

//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    }
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_02) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}); //, nd4j::DataType::DOUBLE);
    auto y = NDArrayFactory::create<double>('c', {2,3}, {3, 3, 3, 3, 3, 3}); //'c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    auto z = NDArrayFactory::create<double>('c', { 2, 3 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 3, 6, 9, 12, 15, 18 });
    //if (x.isActualOnHostSide() && !x.isActualOnDeviceSide())
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
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &y, &z);// *= y;
    //z.lazyAllocateBuffer();
    z.syncToHost();
    z.printBuffer("52Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    //ASSERT_TRUE(exp.equalsTo(z));

//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    }
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_002) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}); //, nd4j::DataType::DOUBLE);
    auto y = NDArrayFactory::create<double>('c', {2, 3}, {2., 3., 3., 3., 3., 3.}); //'c', { 3 }, { 2., 3., 4.}, nd4j::DataType::DOUBLE);
    auto z = NDArrayFactory::create<double>('c', { 2, 3 });

    auto exp = NDArrayFactory::create<double>('c', { 2, 3 }, { 2, 6, 9, 12, 15, 18 });
    //if (x.isActualOnHostSide() && !x.isActualOnDeviceSide())
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
    x.applyPairwiseTransform(pairwise::Multiply, &y, &z);// *= y;
    //z.lazyAllocateBuffer();
    z.syncToHost();
    z.printBuffer("51Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    //ASSERT_TRUE(exp.equalsTo(z));

//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    }
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestBroadcastRaw_1) {

    //if (!Environment::getInstance()->isExperimentalBuild())
    //    return;

    NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);
    NDArray y('c', {3},   {10, 20, 30}, nd4j::DataType::INT64);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);
    NDArray exp('c', {2,3,4}, {10, 11, 12, 13,24, 25, 26, 27,38, 39, 40, 41,22, 23, 24, 25,36, 37, 38, 39,50, 51, 52, 53}, nd4j::DataType::INT32);
    x.linspace(0); x.syncToDevice();

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
    hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(Nd4jLong));							// 0 -- dimensions
    hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
    hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
    std::vector<void*> devicePtrs(hostData.size(), nullptr);

    // create cuda stream and LaunchContext
    cudaError_t cudaResult;
    cudaStream_t* stream = x.getContext()->getCudaStream();
    LaunchContext* pLc = x.getContext();

    // allocate required amount of global device memory and copy host data to it
    //cudaResult = allocateDeviceMem(*pLc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
    for(size_t i = 0; i < devicePtrs.size(); ++i) {
        nd4j_printf("Allocation of %i bytes with device\n", hostData[i].second)
        cudaResult = cudaMalloc(&devicePtrs[i], hostData[i].second); //if(cudaResult != 0) return cudaResult;
        ASSERT_EQ(cudaResult, 0);
        cudaMemcpy(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice);
    }

    // call cuda kernel which calculates result
    NativeOpExecutioner::execBroadcast(pLc, nd4j::broadcast::Add,
                                       nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    cudaResult = cudaStreamSynchronize(*stream); ASSERT_EQ(0, cudaResult);
    z.syncToHost();
    z.printBuffer("ADD broadcasted output");
    // verify results
    for (int e = 0; e < z.lengthOf(); e++)
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

    // free allocated global device memory
    for(int i = 0; i < devicePtrs.size(); ++i)
        cudaFree(devicePtrs[i]);

    // delete cuda stream
    //cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
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
    exp.printBuffer("56Result out");

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
