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
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <graph/Context.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <execution/LaunchContext.h>
#include <ops/specials_cuda.h>
#include <helpers/TAD.h>
#include <ops/declarable/CustomOperations.h>

#include <cuda.h>

using namespace sd;
using namespace sd::graph;

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
    ASSERT_FALSE(x.isActualOnHostSide());
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
    ASSERT_FALSE(x.isActualOnHostSide());

    NDArray::registerSpecialUse({&x}, {&y});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());

    ASSERT_TRUE(y.isActualOnDeviceSide());
    ASSERT_FALSE(y.isActualOnHostSide());
}

TEST_F(NDArrayCudaBasicsTests, Test_Registration_01) {
    auto x = NDArrayFactory::create_<int>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create_<int>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_FALSE(x->isActualOnHostSide());
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
    ASSERT_FALSE(x->isActualOnHostSide());

    NDArray::registerSpecialUse({y}, {x});
    x->applyTransform(transform::Neg, *y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //y->syncToHost();
    // y->printBuffer("Negatives");
    delete x;
    delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_Cosine_1) {
    auto x = NDArrayFactory::create_<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create_<double>('c', {5}, {5, 4, 3, 2, 1});

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_FALSE(x->isActualOnHostSide());

    NDArray::registerSpecialUse({y}, {x});
    x->applyTransform(transform::Cosine, *y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //y->syncToHost();
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
    z.tickWriteDevice();
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);

    for (int e = 0; e < z.lengthOf(); e++)
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestAdd_2) {
    // allocating host-side arrays
    NDArray x('c', { 5 }, { 1, 2, 3, 4, 5});
    NDArray y('c', { 5 }, { 1, 2, 3, 4, 5});
    NDArray z('c', { 5 }, sd::DataType::DOUBLE);

    NDArray exp('c', { 5 }, { 2, 4, 6, 8, 10 });

    Nd4jPointer nativeStream = (Nd4jPointer)malloc(sizeof(cudaStream_t));
    CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));
    cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
    auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    LaunchContext lc(stream, *stream, nullptr, nullptr);
    NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr);
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);

    for (int e = 0; e < z.lengthOf(); e++)
        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
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
    z.tickWriteDevice();
    auto res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);
    //double* localBuffer = ;
    z.syncToHost();
    cudaMemcpy(z.buffer(), z.specialBuffer(), z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost);
    res = cudaStreamSynchronize(*stream);
    z.tickWriteHost();
    ASSERT_EQ(0, res);

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
    x.applyPairwiseTransform(pairwise::Add, y, z);

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
    x.applyPairwiseTransform(pairwise::Multiply, y, z);
    // x.printBuffer("3X = ");
    // y.printBuffer("3Y = ");
    // z.printBuffer("3Result out");

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
    NDArray z('c', { 5 }, sd::DataType::DOUBLE);

    auto exp = NDArrayFactory::create<double>('c', { 5 }, { 1, 4, 9, 16, 25 });

    // making raw buffers
    //Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
    //cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
    //ASSERT_EQ(0, res);
    //res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
    //ASSERT_EQ(0, res);
    x.applyPairwiseTransform(pairwise::Multiply, y, z);

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
    NDArray x('c', { 5 }, { 1, 2, 3, 4, 5}, sd::DataType::DOUBLE);
    NDArray y('c', { 5 }, { 1., 2., 3., 4., 5.}, sd::DataType::DOUBLE);
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
    x.applyPairwiseTransform(pairwise::Multiply, y, z);
    //x.printBuffer("23X = ");
    //y.printBuffer("23Y = ");
    // z.printBuffer("23Result out");

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
    NDArray x('c', { 5 }, { 1, 2, 3, 4, 5}, sd::DataType::DOUBLE);
    NDArray y('c', { 5 }, { 1., 2., 3., 4., 5.}, sd::DataType::DOUBLE);
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
    //x.tickWriteDevice();
    // x.printBuffer("33Result out");

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
    y.tickWriteDevice();

    // x.printBuffer("X = ");
    // y.printBuffer("Y = ");

    for (int e = 0; e < y.lengthOf(); e++) {
        ASSERT_NEAR(exp.e<int>(e), y.e<int>(e), 1e-5);
    }
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveNeg_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());

    x.applyTransform(transform::Neg, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    // y.printBuffer("Negatives2");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveSqrt_1) { // strict
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    auto exp = NDArrayFactory::create<double>({1.000000, 1.414214, 1.732051, 2.000000, 2.236068});
    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());

    x.applyTransform(transform::Sqrt, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    ASSERT_TRUE(y.equalsTo(exp));
    //y.printBuffer("SQRT output");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveAssign_1) { // strict
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    //auto exp = NDArrayFactory::create<double>({1.000000, 1.414214, 1.732051, 2.000000, 2.236068});
    //ASSERT_TRUE(x.isActualOnDeviceSide());
    //ASSERT_TRUE(x.isActualOnHostSide());

    x.applyTransform(transform::Assign, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);

    // printf("Assigned to another array\n");
    // y.printBuffer("OUput");
    ASSERT_TRUE(y.equalsTo(x));
    //y.syncToHost();
    //y.printBuffer("IsMax output");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveCosine_1) { // strict
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    auto exp = NDArrayFactory::create<double>('c', {5}, {0.540302, -0.416147, -0.989992, -0.653644, 0.283662});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());

    x.applyTransform(transform::Cosine, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.dataType() == y.dataType());
    //y.printBuffer("Cosine2");
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveCosine_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    auto exp = NDArrayFactory::create<double>('c', {5}, {0.540302, -0.416147, -0.989992, -0.653644, 0.283662});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());
    x.applyTransform(transform::Cosine, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    //exp.syncToHost();
    //y.printBuffer("PrimitiveCosine2");
    //exp.printBuffer("Primitive Cosine exp");
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.dataType() == y.dataType());
    //for (int e = 0; e < y.lengthOf(); e++) {
    //    ASSERT_NEAR(exp.e<double>(e), y.e<double>(e), 1e-5);
    //}

    ASSERT_TRUE(exp.equalsTo(y));
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, Test_PrimitiveCosine_3) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {5});
    auto exp = NDArrayFactory::create<double>({0.540302, -0.416147, -0.989992, -0.653644, 0.283662});

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());
    x.applyTransform(transform::Cosine, y);
    //ASSERT_TRUE(x->isActualOnDeviceSide());
    //ASSERT_FALSE(x->isActualOnHostSide());

    //ASSERT_TRUE(y->isActualOnDeviceSide());
    //ASSERT_TRUE(y->isActualOnHostSide());
    //auto res = cudaStreamSynchronize(*y.getContext()->getCudaStream());
    //ASSERT_EQ(0, res);
    //exp.syncToHost();
//    y.printBuffer("PrimitiveCosine3");
//    exp.printBuffer("Primitive Cosine3 exp");
//    y.printShapeInfo("Y shape");
//    exp.printShapeInfo("Exp Shape");
    ASSERT_TRUE(exp.isSameShape(y));
//
//    for (int e = 0; e < y.lengthOf(); e++) {
//        printf("%lf == %lf\n", exp.e<double>(e), y.e<double>(e));
////        ASSERT_NEAR(exp.e<double>(e), y.e<double>(e), 1e-5);
//    }

    ASSERT_TRUE(exp.equalsTo(y));
    //delete x;
    //delete y;
}

TEST_F(NDArrayCudaBasicsTests, TestRawBroadcast_2) {

    //if (!Environment::getInstance().isExperimentalBuild())
    //    return;

    NDArray x = NDArrayFactory::create<double>('c', {2,3,4});
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, sd::DataType::DOUBLE);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
//    NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, sd::DataType::DOUBLE);
    NDArray exp('c', {2,3,4}, {10., 40., 90., 160., 50., 120., 210., 320., 90., 200., 330., 480., 650., 840., 1050., 1280., 850., 1080., 1330., 1600., 1050., 1320., 1610., 1920.}, sd::DataType::DOUBLE);
    x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
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
    NativeOpExecutioner::execBroadcast(&lc, sd::broadcast::Multiply,
                                       nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

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

    //if (!Environment::getInstance().isExperimentalBuild())
    //    return;

    NDArray x('c', {2,3,4}, sd::DataType::DOUBLE);
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, sd::DataType::DOUBLE);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
//    NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, sd::DataType::DOUBLE);
    NDArray exp('c', {2,3,4}, {10., 40., 90., 160., 50., 120., 210., 320., 90., 200., 330., 480., 650., 840., 1050., 1280., 850., 1080., 1330., 1600., 1050., 1320., 1610., 1920.}, sd::DataType::DOUBLE);
    x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
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
//    cudaResult = allocateDeviceMem(*pLc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
    for(int i = 0; i < devicePtrs.size(); ++i) {

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&devicePtrs[i]), hostData[i].second); ASSERT_EQ(0, cudaResult);
        cudaMemcpyAsync(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice, *stream);
    }

    NDArray::registerSpecialUse({&z}, {&x, &y});
    // call cuda kernel which calculates result
    NativeOpExecutioner::execBroadcast(pLc, sd::broadcast::Multiply,
                                       nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    //cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    //z.syncToHost();
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
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, sd::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(3.); //'c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    x *= y;
    //x.syncToHost();

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
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, sd::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(3.); //'c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);// *= y;
    // z.printBuffer("53Result out");

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
    auto x = NDArrayFactory::create<double>('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}); //, sd::DataType::DOUBLE);
    auto y = NDArrayFactory::create<double>('c', {2,3}, {3, 3, 3, 3, 3, 3}); //'c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);// *= y;

    // z.printBuffer("52Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    ASSERT_TRUE(exp.equalsTo(z));

//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    }
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply_002) {
    // allocating host-side arrays
    auto x = NDArrayFactory::create<double>('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}); //, sd::DataType::DOUBLE);
    auto y = NDArrayFactory::create<double>('c', {2, 3}, {2., 3., 3., 3., 3., 3.}); //'c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    x.applyPairwiseTransform(pairwise::Multiply, y, z);// *= y;

    // z.printBuffer("51Result out");

    //
    // cudaFree(devBufferPtrX);
    //cudaFree(devBufferPtrZ);
    //cudaFree(devShapePtrX);
    ASSERT_TRUE(exp.equalsTo(z));

//    for (int e = 0; e < x.lengthOf(); e++) {
//        ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    }
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestBroadcastRaw_1) {

    //if (!Environment::getInstance().isExperimentalBuild())
    //    return;

    NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
    NDArray y('c', {3},   {10, 20, 30}, sd::DataType::INT64);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
    NDArray exp('c', {2,3,4}, {10, 11, 12, 13,24, 25, 26, 27,38, 39, 40, 41,22, 23, 24, 25,36, 37, 38, 39,50, 51, 52, 53}, sd::DataType::INT32);
    //real output [10, 11, 12, 13, 4, 5, 6, 7, 28, 29, 30, 31, 22, 23, 24, 25, 16, 17, 18, 19, 40, 41, 42, 43]
    x.linspace(0); x.syncToDevice();

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
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
        cudaResult = cudaMalloc(&devicePtrs[i], hostData[i].second); //if(cudaResult != 0) return cudaResult;
        ASSERT_EQ(cudaResult, 0);
        cudaMemcpy(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice);
    }

    // call cuda kernel which calculates result
    NativeOpExecutioner::execBroadcast(pLc, sd::broadcast::Add,
                                       nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                                       nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                                       nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                       (int*)devicePtrs[0], dimensions.size(),
                                       (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
                                       nullptr, nullptr);

    cudaResult = cudaStreamSynchronize(*stream); ASSERT_EQ(0, cudaResult);

    // x.printIndexedBuffer(" X");
    // y.printIndexedBuffer("+Y");
    // z.printBuffer("ADD broadcasted output");
    // verify results
   // for (int e = 0; e < z.lengthOf(); e++)
   //     ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

    // free allocated global device memory
    for(int i = 0; i < devicePtrs.size(); ++i)
        cudaFree(devicePtrs[i]);

    // delete cuda stream
    //cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

TEST_F(NDArrayCudaBasicsTests, TestBroadcastMultiply) {
    // allocating host-side arrays
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, sd::DataType::DOUBLE);
    NDArray y('c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    NDArray x('c', { 2, 3 }, { 1, 2, 3, 4, 5, 6}, sd::DataType::DOUBLE);
    NDArray y('c', { 3 }, { 2., 3., 4.}, sd::DataType::DOUBLE);
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
    //void NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, ExtraArguments *extraArgs)
    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, exp);

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

    ASSERT_NEAR(y.e<double>(0), 15, 1e-5);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestDup1) {

    NDArray array('c', {2,3}, {1,2,3,4,5,6});
    auto arrC = array.dup('c');
    auto arrF = array.dup('f');
    // arrC->printBuffer("arrC");

    // arrF->printBuffer("arrF");
    //arrC->printShapeInfo("C shape");
    //arrF->printShapeInfo("F shape");

    ASSERT_TRUE(array.equalsTo(arrF));
    ASSERT_TRUE(array.equalsTo(arrC));

    ASSERT_TRUE(arrF.equalsTo(arrC));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, equalsTo_1) {

    NDArray x('c', {2,5}, {1,2,3,4,5,6,7,8,9,10}, sd::DataType::DOUBLE);
    NDArray y('c', {2,5}, {1,2,3,4,5,6,7,8,9,10}, sd::DataType::DOUBLE);

    ASSERT_TRUE(x.equalsTo(y));

    x.permutei({1,0});
    y.permutei({1,0});

    ASSERT_TRUE(x.equalsTo(y));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, equalsTo_2) {

    NDArray x('c', {2,5}, {1,2,3,4,5,6,7,8,10,10}, sd::DataType::DOUBLE);
    NDArray y('c', {2,5}, {1,2,5,4,5,6,7,8,9,10}, sd::DataType::DOUBLE);

    ASSERT_FALSE(x.equalsTo(y));

    x.permutei({1,0});
    y.permutei({1,0});

    ASSERT_FALSE(x.equalsTo(y));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, equalsTo_3) {

    NDArray x('c', {2,5}, {1,2,3,4,5,6,7,8,9,10}, sd::DataType::DOUBLE);
    NDArray y('c', {2,5}, {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f}, sd::DataType::FLOAT32);

    ASSERT_FALSE(x.equalsTo(y));

    x.permutei({1,0});
    y.permutei({1,0});

    ASSERT_FALSE(x.equalsTo(y));
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyReduce3_1) {

    NDArray x('c', {2,3,4}, {-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13}, sd::DataType::INT32);
    NDArray x2('c', {2,3,4}, {-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13}, sd::DataType::INT32);
    NDArray y('c', {2,3,4}, {-2,3,-4,5,-2,3,-4,5,-2,3,-4,5,-2,3,-4,5,-2,3,-4,5,-2,3,-4,5}, sd::DataType::INT32);
    NDArray k('c', {2,3}, {-2,3,-4,5,-2,3}, sd::DataType::INT32);
    NDArray k2('c', {3,2}, {-2,3,-4,5,-2,3}, sd::DataType::INT32);

    NDArray exp1('c', {3}, {4.f, 20.f, 36.f}, sd::DataType::FLOAT32);
    NDArray exp2('c', {2,3}, {-10.f, -2.f, 6.f,14.f, 22.f, 30.f}, sd::DataType::FLOAT32);
    NDArray exp3('c', {4}, {38.f, 41.f, 44.f, 47.f}, sd::DataType::FLOAT32);
    NDArray exp4('c', {4}, {114.f, 117.f, 120.f, 123.f}, sd::DataType::FLOAT32);


    NDArray z = x.applyReduce3(sd::reduce3::Dot, y, {0,2});
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x.applyReduce3(sd::reduce3::Dot, k, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp3));

    x.permutei({0,2,1});
    y.permutei({0,2,1});

    z = y.applyReduce3(sd::reduce3::Dot, x, {1});
    ASSERT_TRUE(z.equalsTo(&exp2));

    x2.permutei({1,0,2});

    z = x2.applyReduce3(sd::reduce3::Dot, k2, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp4));
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyReduce3_2) {

    NDArray x('c', {2,3,4}, {-10,-9,-8.5,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13}, sd::DataType::DOUBLE);
    NDArray x2('c', {2,3,4}, {-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13}, sd::DataType::DOUBLE);
    NDArray y('c', {2,3,4}, {-2,3,-4,5,-2,3,-4,5,-2,3,-4,5,-2.5,3,-4,5,-2,3,-4,5,-2,3,-4,5}, sd::DataType::DOUBLE);
    NDArray k('c', {2,3}, {-2,3,-4,5.5,-2,3}, sd::DataType::DOUBLE);
    NDArray k2('c', {3,2}, {-2,3,-4,5,-2,3.5}, sd::DataType::DOUBLE);

    NDArray exp1('c', {3}, {5., 20., 36.}, sd::DataType::DOUBLE);
    NDArray exp2('c', {2,3}, {-8., -2., 6., 13., 22., 30.}, sd::DataType::DOUBLE);
    NDArray exp3('c', {4}, {39., 42.5, 47., 49.5}, sd::DataType::DOUBLE);
    NDArray exp4('c', {4}, {119., 122.5, 125., 129.5}, sd::DataType::DOUBLE);

    NDArray z = x.applyReduce3(sd::reduce3::Dot, y, {0,2});
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x.applyReduce3(sd::reduce3::Dot, k, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp3));

    x.permutei({0,2,1});
    y.permutei({0,2,1});

    z = y.applyReduce3(sd::reduce3::Dot, x, {1});
    ASSERT_TRUE(z.equalsTo(&exp2));

    x2.permutei({1,0,2});

    z = x2.applyReduce3(sd::reduce3::Dot, k2, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp4));
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyReduce3_3) {

    NDArray x1('c', {2,2,2}, {1,2,3,4,5,6,7,8}, sd::DataType::INT32);
    NDArray x2('c', {2,2,2}, {-1,-2,-3,-4,-5,-6,-7,-8}, sd::DataType::INT32);
    NDArray x3('c', {3,2}, {1.5,1.5,1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);
    NDArray x4('c', {3,2}, {1,2,3,4,5,6}, sd::DataType::DOUBLE);

    NDArray exp1('c', {}, std::vector<double>{-204}, sd::DataType::FLOAT32);
    NDArray exp2('c', {}, std::vector<double>{31.5}, sd::DataType::DOUBLE);


    auto z = x1.applyReduce3(reduce3::Dot, x2);
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x3.applyReduce3(reduce3::Dot, x4);
    ASSERT_TRUE(z.equalsTo(&exp2));

    x1.permutei({2,1,0});
    x2.permutei({2,1,0});
    x3.permutei({1,0});
    x4.permutei({1,0});

    z = x1.applyReduce3(reduce3::Dot, x2);
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x3.applyReduce3(reduce3::Dot, x4);
    ASSERT_TRUE(z.equalsTo(&exp2));
}

////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyAllReduce3_1) {

    NDArray x1('c', {2,3,2}, {1,2,3,4,5,6,7,8,-1,-2,-3,-4,}, sd::DataType::INT32);
    NDArray x2('c', {2,2,2}, {-1,-2,-3,-4,-5,-6,-7,-8}, sd::DataType::INT32);
    NDArray x3('c', {3,2}, {1.5,1.5,1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);
    NDArray x4('c', {3,2}, {1,2,3,4,5,6}, sd::DataType::DOUBLE);

    NDArray exp1('c', {3,2}, {-88.f, -124.f, 6.f, -2.f, 22.f, 14.f}, sd::DataType::FLOAT32);
    NDArray exp2('c', {6,4}, {-36.f, -44.f, -52.f, -60.f,-42.f, -52.f, -62.f, -72.f, 2.f, 0.f, -2.f,
                              -4.f, 6.f, 4.f, 2.f, 0.f, 10.f, 8.f, 6.f, 4.f, 14.f, 12.f, 10.f, 8.f},
            sd::DataType::FLOAT32);
    NDArray exp3('c', {1,1}, std::vector<double>{31.5}, sd::DataType::DOUBLE);
    NDArray exp4('c', {3,3}, {4.5, 10.5, 16.5,4.5, 10.5, 16.5,4.5, 10.5, 16.5}, sd::DataType::DOUBLE);

    auto z = x1.applyAllReduce3(reduce3::Dot, x2, {0,2});
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x1.applyAllReduce3(reduce3::Dot, x2, {0});
    ASSERT_TRUE(z.equalsTo(&exp2));

    z = x3.applyAllReduce3(reduce3::Dot, x4, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp3));

    z = x3.applyAllReduce3(reduce3::Dot, x4, {1});
    ASSERT_TRUE(z.equalsTo(&exp4));

    x1.permutei({2,1,0});
    x2.permutei({2,1,0});
    x3.permutei({1,0});
    x4.permutei({1,0});

    z = x1.applyAllReduce3(reduce3::Dot, x2, {0,2});
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x3.applyAllReduce3(reduce3::Dot, x4, {0});
    ASSERT_TRUE(z.equalsTo(&exp4));
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyIndexReduce_test1) {

    NDArray x('c', {2,3}, {0, 10, 1, 2, 2.5,-4}, sd::DataType::DOUBLE);

    NDArray scalar('c', {}, std::vector<double>{100}, sd::DataType::INT64);
    NDArray vec1('c', {2}, {100,100}, sd::DataType::INT64);
    NDArray vec2('c', {3}, {100,100,100}, sd::DataType::INT64);

    NDArray exp1('c', {}, std::vector<double>{1}, sd::DataType::INT64);
    NDArray exp2('c', {2}, {1,1}, sd::DataType::INT64);
    NDArray exp3('c', {3}, {1,0,0}, sd::DataType::INT64);

    NDArray exp4('c', {}, std::vector<double>{2}, sd::DataType::INT64);
    NDArray exp5('c', {2}, {1,1}, sd::DataType::INT64);
    NDArray exp6('c', {3}, {1,0,0}, sd::DataType::INT64);

    x.applyIndexReduce(sd::indexreduce::IndexMax, scalar, {0,1});
    ASSERT_TRUE(scalar.equalsTo(&exp1));

    x.applyIndexReduce(sd::indexreduce::IndexMax, vec1, {1});
    ASSERT_TRUE(vec1.equalsTo(&exp2));

    x.applyIndexReduce(sd::indexreduce::IndexMax, vec2, {0});
    ASSERT_TRUE(vec2.equalsTo(&exp3));

    x.permutei({1,0});

    x.applyIndexReduce(sd::indexreduce::IndexMax, scalar, {0,1});
    ASSERT_TRUE(scalar.equalsTo(&exp4));

    x.applyIndexReduce(sd::indexreduce::IndexMax, vec1, {0});
    ASSERT_TRUE(vec1.equalsTo(&exp5));

    x.applyIndexReduce(sd::indexreduce::IndexMax, vec2, {1});
    ASSERT_TRUE(vec2.equalsTo(&exp6));
}


//////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, applyIndexReduce_test2) {

    NDArray x('c', {2,3}, {0, 10, 1, 2, 2.5,-4}, sd::DataType::DOUBLE);

    NDArray exp1('c', {}, std::vector<double>{1}, sd::DataType::INT64);
    NDArray exp2('c', {2}, {1,1}, sd::DataType::INT64);
    NDArray exp3('c', {3}, {1,0,0}, sd::DataType::INT64);

    NDArray exp4('c', {}, std::vector<double>{2}, sd::DataType::INT64);
    NDArray exp5('c', {2}, {1,1}, sd::DataType::INT64);
    NDArray exp6('c', {3}, {1,0,0}, sd::DataType::INT64);

    auto z = x.applyIndexReduce(sd::indexreduce::IndexMax, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp1));

    z = x.applyIndexReduce(sd::indexreduce::IndexMax, {1});
    ASSERT_TRUE(z.equalsTo(&exp2));

    z = x.applyIndexReduce(sd::indexreduce::IndexMax, {0});
    ASSERT_TRUE(z.equalsTo(&exp3));

    x.permutei({1,0});

    z = x.applyIndexReduce(sd::indexreduce::IndexMax, {0,1});
    ASSERT_TRUE(z.equalsTo(&exp4));

    z = x.applyIndexReduce(sd::indexreduce::IndexMax, {0});
    ASSERT_TRUE(z.equalsTo(&exp5));

    z = x.applyIndexReduce(sd::indexreduce::IndexMax, {1});
    ASSERT_TRUE(z.equalsTo(&exp6));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_float_test1) {

    NDArray x('c', {2,3,2}, {1,2,3,4,5,6,7,8,-1,-2,-3,-4,}, sd::DataType::INT32);

    NDArray z1('c', {}, std::vector<double>{100}, sd::DataType::DOUBLE);
    NDArray z2('c', {2,2}, {100,100,100,100}, sd::DataType::FLOAT32);
    NDArray z3('c', {3}, {100,100,100}, sd::DataType::DOUBLE);
    NDArray z4('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);
    NDArray z5('c', {2}, {100,100}, sd::DataType::FLOAT32);

    NDArray exp1('c', {}, std::vector<double>{2.166667}, sd::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {3.f,4.f,1.f,0.666667f}, sd::DataType::FLOAT32);
    NDArray exp3('c', {3}, {4.5,1,1}, sd::DataType::DOUBLE);
    NDArray exp4('c', {3,2}, {4,5,1,1,1,1}, sd::DataType::FLOAT32);
    NDArray exp5('c', {2}, {3.5f,0.833333f}, sd::DataType::FLOAT32);

    x.reduceAlongDimension(sd::reduce::Mean, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::Mean, z2, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    x.reduceAlongDimension(sd::reduce::Mean, z3, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    x.reduceAlongDimension(sd::reduce::Mean, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::Mean, z4, {1});
    ASSERT_TRUE(z4.equalsTo(&exp4));

    x.reduceAlongDimension(sd::reduce::Mean, z5, {0,2});
    ASSERT_TRUE(z5.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_float_test2) {

    NDArray x('c', {2,3,2}, {1,2,3,4,5,6,7,8,-1,-2,-3,-4,}, sd::DataType::DOUBLE);

    NDArray exp1('c', {}, std::vector<double>{2.166667}, sd::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {3,4,1,0.666667}, sd::DataType::DOUBLE);
    NDArray exp3('c', {3}, {4.5,1,1}, sd::DataType::DOUBLE);
    NDArray exp4('c', {3,2}, {4,5,1,1,1,1}, sd::DataType::DOUBLE);
    NDArray exp5('c', {2}, {3.5,0.833333}, sd::DataType::DOUBLE);

    NDArray z1 = x.reduceAlongDimension(sd::reduce::Mean, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    NDArray z2 = x.reduceAlongDimension(sd::reduce::Mean, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    NDArray z3 = x.reduceAlongDimension(sd::reduce::Mean, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    NDArray z4 = x.reduceAlongDimension(sd::reduce::Mean, {0,1,2});
    ASSERT_TRUE(z4.equalsTo(&exp1));

    NDArray z5 = x.reduceAlongDimension(sd::reduce::Mean, {1});
    ASSERT_TRUE(z5.equalsTo(&exp4));

    NDArray z6 = x.reduceAlongDimension(sd::reduce::Mean, {0,2});
    ASSERT_TRUE(z6.equalsTo(&exp5));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, EqualityTest1) {
    auto arrayA = NDArrayFactory::create_<float>('f', {3, 5});
    auto arrayB = NDArrayFactory::create_<float>('f', {3, 5});
    auto arrayC = NDArrayFactory::create_<float>('f', {3, 5});

    auto arrayD = NDArrayFactory::create_<float>('f', {2, 4});
    auto arrayE = NDArrayFactory::create_<float>('f', {1, 15});

    for (int i = 0; i < arrayA->rows(); i++) {
        for (int k = 0; k < arrayA->columns(); k++) {
            arrayA->p(i, k, (float) i);
        }
    }

    for (int i = 0; i < arrayB->rows(); i++) {
        for (int k = 0; k < arrayB->columns(); k++) {
            arrayB->p(i, k, (float) i);
        }
    }

    for (int i = 0; i < arrayC->rows(); i++) {
        for (int k = 0; k < arrayC->columns(); k++) {
            arrayC->p(i, k, (float) i+1);
        }
    }

    ASSERT_TRUE(arrayA->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayC->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayD->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayE->equalsTo(arrayB, 1e-5));

    delete arrayA;
    delete arrayB;
    delete arrayC;
    delete arrayD;
    delete arrayE;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_same_test1) {

    NDArray x('c', {2,3,2}, {1.5f,2.f,3.f,4.f,5.f,6.f,7.5f,8.f,-1.f,-2.f,-3.5f,-4.f}, sd::DataType::FLOAT32);

    NDArray z1('c', {}, std::vector<double>{100}, sd::DataType::FLOAT32);
    NDArray z2('c', {2,2}, {100,100,100,100}, sd::DataType::FLOAT32);
    NDArray z3('c', {3}, {100,100,100}, sd::DataType::FLOAT32);
    NDArray z4('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);
    NDArray z5('c', {2}, {100,100}, sd::DataType::FLOAT32);

    NDArray exp1('c', {}, std::vector<double>{26.5f}, sd::DataType::FLOAT32);
    NDArray exp2('c', {2,2}, {9.5f,12.f,3.f,2.f}, sd::DataType::FLOAT32);
    NDArray exp3('c', {3}, {19.f,4.f,3.5f}, sd::DataType::FLOAT32);
    NDArray exp4('c', {3,2}, {9.f,10.f,2.f,2.f,1.5f,2.f}, sd::DataType::FLOAT32);
    NDArray exp5('c', {2}, {21.5f,5.f}, sd::DataType::FLOAT32);

    x.reduceAlongDimension(sd::reduce::Sum, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::Sum, z2, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    x.reduceAlongDimension(sd::reduce::Sum, z3, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    x.reduceAlongDimension(sd::reduce::Sum, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::Sum, z4, {1});
    ASSERT_TRUE(z4.equalsTo(&exp4));

    x.reduceAlongDimension(sd::reduce::Sum, z5, {0,2});
    ASSERT_TRUE(z5.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_same_test2) {

    NDArray x('c', {2,3,2}, {1.5,2,3,4,5,6,7.5,8,-1,-2,-3.5,-4,}, sd::DataType::INT64);

    NDArray exp1('c', {}, std::vector<double>{26}, sd::DataType::INT64);
    NDArray exp2('c', {2,2}, {9,12,3,2}, sd::DataType::INT64);
    NDArray exp3('c', {3}, {18,4,4}, sd::DataType::INT64);
    NDArray exp4('c', {3,2}, {8,10,2,2,2,2}, sd::DataType::INT64);
    NDArray exp5('c', {2}, {21,5}, sd::DataType::INT64);

    NDArray z1 = x.reduceAlongDimension(sd::reduce::Sum, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    NDArray z2 = x.reduceAlongDimension(sd::reduce::Sum, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    NDArray z3 = x.reduceAlongDimension(sd::reduce::Sum, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    NDArray z4 = x.reduceAlongDimension(sd::reduce::Sum, {0,1,2});
    ASSERT_TRUE(z4.equalsTo(&exp1));

    NDArray z5 = x.reduceAlongDimension(sd::reduce::Sum, {1});
    ASSERT_TRUE(z5.equalsTo(&exp4));

    NDArray z6 = x.reduceAlongDimension(sd::reduce::Sum, {0,2});
    ASSERT_TRUE(z6.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_bool_test1) {

    NDArray x('c', {2,3,2}, {0.5,2,3,-4,5,6,-7.5,8,-1,-0.5,-3.5,4}, sd::DataType::DOUBLE);

    NDArray z1('c', {}, std::vector<double>{true}, sd::DataType::BOOL);
    NDArray z2('c', {2,2}, {true,true,true,true}, sd::DataType::BOOL);
    NDArray z3('c', {3}, {true,true,true}, sd::DataType::BOOL);
    NDArray z4('c', {3,2}, {true,true,true,true,true,true}, sd::DataType::BOOL);
    NDArray z5('c', {2}, {true,true}, sd::DataType::BOOL);

    NDArray exp1('c', {}, std::vector<double>{true}, sd::DataType::BOOL);
    NDArray exp2('c', {2,2}, {true,true,false,true}, sd::DataType::BOOL);
    NDArray exp3('c', {3}, {true,true,true}, sd::DataType::BOOL);
    NDArray exp4('c', {3,2}, {true,true,true,false,true,true}, sd::DataType::BOOL);
    NDArray exp5('c', {2}, {true,true}, sd::DataType::BOOL);

    x.reduceAlongDimension(sd::reduce::IsPositive, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::IsPositive, z2, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    x.reduceAlongDimension(sd::reduce::IsPositive, z3, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    x.reduceAlongDimension(sd::reduce::IsPositive, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::IsPositive, z4, {1});
    ASSERT_TRUE(z4.equalsTo(&exp4));

    x.reduceAlongDimension(sd::reduce::IsPositive, z5, {0,2});
    ASSERT_TRUE(z5.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_bool_test2) {

    NDArray x('c', {2,3,2}, {0.5,2,3,-4,5,6,-7.5,8,-1,-0.5,-3.5,4}, sd::DataType::INT32);

    NDArray exp1('c', {}, std::vector<double>{1}, sd::DataType::BOOL);
    NDArray exp2('c', {2,2}, {1,1,0,1}, sd::DataType::BOOL);
    NDArray exp3('c', {3}, {1,1,1}, sd::DataType::BOOL);
    NDArray exp4('c', {3,2}, {0,1,1,0,1,1}, sd::DataType::BOOL);
    NDArray exp5('c', {2}, {1,1}, sd::DataType::BOOL);

    NDArray z1 = x.reduceAlongDimension(sd::reduce::IsPositive, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    NDArray z2 = x.reduceAlongDimension(sd::reduce::IsPositive, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    NDArray z3 = x.reduceAlongDimension(sd::reduce::IsPositive, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    NDArray z4 = x.reduceAlongDimension(sd::reduce::IsPositive, {0,1,2});
    ASSERT_TRUE(z4.equalsTo(&exp1));

    NDArray z5 = x.reduceAlongDimension(sd::reduce::IsPositive, {1});
    ASSERT_TRUE(z5.equalsTo(&exp4));

    NDArray z6 = x.reduceAlongDimension(sd::reduce::IsPositive, {0,2});
    ASSERT_TRUE(z6.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_long_test1) {

    NDArray x('c', {2,3,2}, {0.5f,2.f,3.f,-0.f,5.f,6.f,-7.5f,0.f,-1.f,-0.5f,-3.5f,4.f}, sd::DataType::FLOAT32);

    NDArray z1('c', {}, std::vector<double>{100}, sd::DataType::INT64);
    NDArray z2('c', {2,2}, {100,100,100,100}, sd::DataType::INT64);
    NDArray z3('c', {3}, {100,100,100}, sd::DataType::INT64);
    NDArray z4('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::INT64);
    NDArray z5('c', {2}, {100,100}, sd::DataType::INT64);

    NDArray exp1('c', {}, std::vector<double>{2}, sd::DataType::INT64);
    NDArray exp2('c', {2,2}, {0,1,0,1}, sd::DataType::INT64);
    NDArray exp3('c', {3}, {1,1,0}, sd::DataType::INT64);
    NDArray exp4('c', {3,2}, {0,1,0,1,0,0}, sd::DataType::INT64);
    NDArray exp5('c', {2}, {1,1}, sd::DataType::INT64);

    x.reduceAlongDimension(sd::reduce::CountZero, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::CountZero, z2, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    x.reduceAlongDimension(sd::reduce::CountZero, z3, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    x.reduceAlongDimension(sd::reduce::CountZero, z1, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    x.reduceAlongDimension(sd::reduce::CountZero, z4, {1});
    ASSERT_TRUE(z4.equalsTo(&exp4));

    x.reduceAlongDimension(sd::reduce::CountZero, z5, {0,2});
    ASSERT_TRUE(z5.equalsTo(&exp5));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, reduceAlongDimension_long_test2) {

    NDArray x('c', {2,3,2}, {0.5,2,3,-0,5,6,-7.5,0,-1,-0.5,-3.5,4}, sd::DataType::INT32);

    NDArray exp1('c', {}, std::vector<double>{4}, sd::DataType::INT64);
    NDArray exp2('c', {2,2}, {1,1,0,2}, sd::DataType::INT64);
    NDArray exp3('c', {3}, {2,2,0}, sd::DataType::INT64);
    NDArray exp4('c', {3,2}, {1,1,0,2,0,0}, sd::DataType::INT64);
    NDArray exp5('c', {2}, {2,2}, sd::DataType::INT64);

    NDArray z1 = x.reduceAlongDimension(sd::reduce::CountZero, {0,1,2});
    ASSERT_TRUE(z1.equalsTo(&exp1));

    NDArray z2 = x.reduceAlongDimension(sd::reduce::CountZero, {1});
    ASSERT_TRUE(z2.equalsTo(&exp2));

    NDArray z3 = x.reduceAlongDimension(sd::reduce::CountZero, {0,2});
    ASSERT_TRUE(z3.equalsTo(&exp3));

    x.permutei({1,0,2});    // 3x2x2

    NDArray z4 = x.reduceAlongDimension(sd::reduce::CountZero, {0,1,2});
    ASSERT_TRUE(z4.equalsTo(&exp1));

    NDArray z5 = x.reduceAlongDimension(sd::reduce::CountZero, {1});
    ASSERT_TRUE(z5.equalsTo(&exp4));

    NDArray z6 = x.reduceAlongDimension(sd::reduce::CountZero, {0,2});
    ASSERT_TRUE(z6.equalsTo(&exp5));
}

TEST_F(NDArrayCudaBasicsTests, BroadcastOpsTest1) {

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto z = NDArrayFactory::create<float>('c', {5, 5});
    auto row = NDArrayFactory::linspace(1.0f, 5.0f, 5);
    NDArray expRow('c', {1, 5,}, {1,2,3,4,5}, sd::DataType::FLOAT32);
    NDArray exp('c', {5,5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, sd::DataType::FLOAT32);

    ASSERT_TRUE(row->equalsTo(&expRow));

    x.applyBroadcast(broadcast::Add, {1}, *row, z);
    x += *row;

    ASSERT_TRUE(x.equalsTo(z));
    //ASSERT_TRUE(z.equalsTo(&exp));

    delete row;
}

TEST_F(NDArrayCudaBasicsTests, BroadcastOpsTest2) {

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    //auto z = NDArrayFactory::create<float>('c', {5, 5});
    auto row = NDArrayFactory::linspace(1.0f, 5.0f, 5);
    NDArray expRow('c', {1, 5,}, {1,2,3,4,5}, sd::DataType::FLOAT32);
    NDArray exp('c', {5,5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, sd::DataType::FLOAT32);

    ASSERT_TRUE(row->equalsTo(&expRow));
    x.applyBroadcast(broadcast::Add, {1}, *row, x);
    ASSERT_TRUE(x.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, TestBroadcast_1) {

    NDArray exp('c', {2, 3, 2, 2}, {1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.}, sd::DataType::DOUBLE);

    auto input = NDArrayFactory::create<double>('c',{ 2, 3, 2, 2});
    auto bias = NDArrayFactory::create<double>('c', {1, 3});

    bias.linspace(1);
    input.applyBroadcast(broadcast::Add, {1}, bias, input);
    ASSERT_TRUE(exp.equalsTo(&input));
}

TEST_F(NDArrayCudaBasicsTests, TestFloat16_1) {
    auto x = NDArrayFactory::create<float>({1,2,3,4,5,7,8,9});
    auto y = NDArrayFactory::create<float>({1,2,3,4,5,7,8,9});
    ASSERT_TRUE(x.equalsTo(&y));
}

TEST_F(NDArrayCudaBasicsTests, TestFloat16_2) {
    auto x = NDArrayFactory::create<float16>('c', {9}, {1,2,3,4,5,6,7,8,9});
    auto y = NDArrayFactory::create<float16>('c', {9}, {1,2,3,4,5,6,7,8,9});
    ASSERT_TRUE(x.equalsTo(y));
    //for (int e = 0; e < x.lengthOf(); e++)
    //    ASSERT_NEAR(x.e<float16>(e), y.e<float16>(e), 1.e-5f);
}

TEST_F(NDArrayCudaBasicsTests, TestFloat16_3) {
    auto x = NDArrayFactory::create<bfloat16>({1,2,3,4,5,7,8,9});
    auto y = NDArrayFactory::create<bfloat16>({1,2,3,4,5,7,8,9});
    ASSERT_TRUE(x.equalsTo(&y));
}

TEST_F(NDArrayCudaBasicsTests, TestFloat_4) {
    auto x = NDArrayFactory::create<float>({1,2,3,4,5,7,8,9});
    auto y = NDArrayFactory::create<float>({2,4,5,5,6,7,8,9});
    ASSERT_FALSE(x.equalsTo(&y));
}

TEST_F(NDArrayCudaBasicsTests, TestFloat_5) {
    auto x = NDArrayFactory::create<float>('c', {3,3}, {1,2,3,4,5,6,7,8,9});
    auto y = NDArrayFactory::create<float>('c', {3,3}, {2,4,5,5,6,7,8,9, 10});
    ASSERT_FALSE(x.equalsTo(&y));
}

TEST_F(NDArrayCudaBasicsTests, TestFloat_6) {
    auto x = NDArrayFactory::create<float>('f', {3,3}, {1,2,3,4,5,6,7,8,9});
    auto y = NDArrayFactory::create<float>('f', {3,3}, {2,4,5,5,6,7,8,9,10});
    ASSERT_FALSE(x.equalsTo(&y));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, Operator_Plus_Test_05)
{
    auto x = NDArrayFactory::create<float>('c', {8, 8, 8});
    auto y = NDArrayFactory::create<float>('c', {1, 8, 8});
    auto expected = NDArrayFactory::create<float>('c', {8, 8, 8});
    NDArray res2  = NDArrayFactory::create<float>(expected.ordering(), expected.getShapeAsVector());
    x = 1.;
    y = 2.;
    expected = 3.;
    res2 = 0.f;

    x.applyTrueBroadcast(BroadcastOpsTuple::Add(), y, res2);// *= y;

    ASSERT_TRUE(expected.isSameShape(&res2));
    ASSERT_TRUE(expected.equalsTo(&res2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, Operator_Plus_Test_5)
{
    auto x = NDArrayFactory::create<float>('c', {8, 8, 8});
    auto y = NDArrayFactory::create<float>('c', {8, 1, 8});
    auto expected = NDArrayFactory::create<float>('c', {8, 8, 8});
    NDArray res2(expected);
    x = 1.;
    y = 2.;
    expected = 3.;
    //x.printBuffer("X=");
    //y.printBuffer("Y=");
    //expected.printBuffer("EXPECTED");
    auto result = x + y;
    //result.printBuffer("1 + 2 =");
    //res2.assign(x + y);

    //x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &res2);
    //res2.printBuffer("Z=");
    //x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &res2);// *= y;
//    x += y;
    //x.printBuffer("OutputX");
    //res2.syncToHost();
    //res2.printBuffer("OUputZ");
    //x.printIndexedBuffer("OUtputX");
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, Operator_Plus_Test_51)
{
    auto x = NDArrayFactory::create<float>('c', {8, 8, 8});
    auto y = NDArrayFactory::create<float>('c', {8, 8});
    auto expected = NDArrayFactory::create<float>('c', {8, 8, 8});
    NDArray res2(expected);
    x = 1.;
    y = 2.;
    expected = 3.;
    //x.printBuffer("X=");
    //y.printBuffer("Y=");
    //expected.printBuffer("EXPECTED");
    auto result = x + y;
    //result.printBuffer("1 + 2 =");
    //res2.assign(x + y);

    //x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &res2);
    //res2.printBuffer("Z=");
    //x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &res2);// *= y;
//    x += y;
    //x.printBuffer("OutputX");
    //res2.syncToHost();
    //res2.printBuffer("OUputZ");
    //x.printIndexedBuffer("OUtputX");
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

TEST_F(NDArrayCudaBasicsTests, Tile_Test_2_1)
{
    auto x = NDArrayFactory::create<float>('c', {2, 1, 2});
    x = 10.;
    auto y = x.tile({1,2,1});
    auto exp = NDArrayFactory::create<float>('c', {2, 2, 2});
    exp = 10.;

    // y.printShapeInfo("Output SHAPE");
    // y.printBuffer("Output TILE");
    // exp.printBuffer("Expect TILE");
    ASSERT_TRUE(exp.equalsTo(y));
}

TEST_F(NDArrayCudaBasicsTests, Tile_Test_2_2)
{
    auto x = NDArrayFactory::create<float>('f', {2, 1, 2});
    x = 10.;
    auto y = x.tile({1,2,1});
    auto exp = NDArrayFactory::create<float>('f', {2, 2, 2});
    exp = 10.;
    ASSERT_TRUE(exp.equalsTo(y));
}

TEST_F(NDArrayCudaBasicsTests, Tile_Test_2_3)
{
    auto x = NDArrayFactory::create<float>('f', {2, 1, 2});
    x = 10.;
    x.p(1,0,1, 20);
    x.syncToDevice();
    auto y = x.tile({1,2,1});
    auto exp = NDArrayFactory::create<float>('f', {2, 2, 2});
    exp = 10.;
    exp.p(1,0,1, 20.);
    exp.p(1, 1, 1, 20.);
    exp.syncToDevice();
    ASSERT_TRUE(exp.equalsTo(y));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, Operator_Plus_Test_2)
{
    double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};
    NDArray a('c', {4,4}, {1,2,3,4,5,6,7,8,9,2,3,2,1,0,4,7}, sd::DataType::FLOAT32);
    auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
    auto y = NDArrayFactory::create<double>('c',    {1, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

    x.linspace(1);
    y.linspace(1);
    auto result = x + y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, assign_2)
{
    NDArray x('c', {4}, {1.5f,2.5f,3.5f,4.5f}, sd::DataType::FLOAT32);
    NDArray y('c', {4}, sd::DataType::INT32);
    NDArray expected('c', {4}, {1,2,3,4}, sd::DataType::INT32);

    y.assign(x);
    // y.printBuffer("ASSIGN VECTOR");

    ASSERT_TRUE(expected.equalsTo(&y));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, subarray_1)
{
    NDArray x('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, sd::DataType::FLOAT32);
    NDArray y('f', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, sd::DataType::FLOAT32);

    Nd4jLong shapeExpX0[] = {1, 2, 12, 8192, 1, 99};
    float    buffExpX0[]  = {1.f, 13.f};
    Nd4jLong shapeExpX1[] = {1, 2, 12, 8192, 1, 99};
    float    buffExpX1[]  = {2.f, 14.f};
    Nd4jLong shapeExpX2[] = {3, 2, 1, 1, 12, 4, 1, 8192, 1, 99};
    float    buffExpX2[]  = {1.f, 13.f};
    Nd4jLong shapeExpX3[] = {2, 2, 4, 12, 1, 8192, 1, 99};
    float    buffExpX3[]  = {9.f, 10.f, 11.f, 12.f, 21.f, 22.f, 23.f, 24.f};
    Nd4jLong shapeExpX4[] = {3, 2, 1, 4, 12, 4, 1, 8192, 1, 99};
    float    buffExpX4[]  = {9.f, 10.f, 11.f, 12.f, 21.f, 22.f, 23.f, 24.f};
    Nd4jLong shapeExpX5[] = {2, 2, 3, 12, 4, 8192, 1, 99};
    float    buffExpX5[]  = {4.f, 8.f, 12.f, 16.f, 20.f, 24.f};

    Nd4jLong shapeExpY0[] = {1, 2, 1, 8192, 1, 99};
    float    buffExpY0[]  = {1.f, 2.f};
    Nd4jLong shapeExpY1[] = {1, 2, 1, 8192, 1, 99};
    float    buffExpY1[]  = {7.f, 8.f};
    Nd4jLong shapeExpY2[] = {3, 2, 1, 1, 1, 2, 6, 8192, 1, 102};
    float    buffExpY2[]  = {1.f, 2.f};
    Nd4jLong shapeExpY3[] = {2, 2, 4, 1, 6, 8192, 1, 99};
    float    buffExpY3[]  = {5.f, 11.f, 17.f, 23.f, 6.f, 12.f, 18.f, 24.f};
    Nd4jLong shapeExpY4[] = {3, 2, 1, 4, 1, 2, 6, 8192, 1, 102};
    float    buffExpY4[]  = {5.f, 11.f, 17.f, 23.f, 6.f, 12.f, 18.f, 24.f};
    Nd4jLong shapeExpY5[] = {2, 2, 3, 1, 2, 8192, 1, 99};
    float    buffExpY5[]  = {19.f, 21.f, 23.f, 20.f, 22.f, 24.f};


    NDArray x0 = x(0, {1,2});
    NDArray xExp(buffExpX0, shapeExpX0);

    ASSERT_TRUE(xExp.isSameShape(x0));
    ASSERT_TRUE(xExp.equalsTo(x0));
//    for(int i = 0; i < shape::shapeInfoLength(x0.rankOf()); ++i)
//        ASSERT_TRUE(x0.shapeInfo()[i] == shapeExpX0[i]);
//    for(int i = 0; i < x0.lengthOf(); ++i)
//        ASSERT_TRUE(x0.e<float>(i) == buffExpX0[i]);

    NDArray x1 = x(1, {1,2});
    NDArray x1Exp(buffExpX1, shapeExpX1);
    ASSERT_TRUE(x1Exp.isSameShape(x1));
    ASSERT_TRUE(x1Exp.equalsTo(x1));

//    for(int i = 0; i < shape::shapeInfoLength(x1.rankOf()); ++i)
//        ASSERT_TRUE(x1.shapeInfo()[i] == shapeExpX1[i]);
//    for(int i = 0; i < x1.lengthOf(); ++i)
//        ASSERT_TRUE(x1.e<float>(i) == buffExpX1[i]);

    NDArray x2 = x(0, {1,2}, true);
    NDArray x2Exp(buffExpX2, shapeExpX2);
    ASSERT_TRUE(x2Exp.isSameShape(x2));
//    x2.printBuffer("X2");
//    x2Exp.printBuffer("X2 EXPECT");
    ASSERT_TRUE(x2Exp.equalsTo(x2));
//    for(int i = 0; i < shape::shapeInfoLength(x2.rankOf()); ++i)
//        ASSERT_TRUE(x2.shapeInfo()[i] == shapeExpX2[i]);
//    for(int i = 0; i < x2.lengthOf(); ++i)
//        ASSERT_TRUE(x2.e<float>(i) == buffExpX2[i]);

    NDArray x3 = x(2, {1});
    NDArray x3Exp(buffExpX3, shapeExpX3);
    ASSERT_TRUE(x3Exp.isSameShape(x3));
    ASSERT_TRUE(x3Exp.equalsTo(x3));
//    for(int i = 0; i < shape::shapeInfoLength(x3.rankOf()); ++i)
//        ASSERT_TRUE(x3.shapeInfo()[i] == shapeExpX3[i]);
//    for(int i = 0; i < x3.lengthOf(); ++i)
//        ASSERT_TRUE(x3.e<float>(i) == buffExpX3[i]);

    NDArray x4 = x(2, {1}, true);
    NDArray x4Exp(buffExpX4, shapeExpX4);
    ASSERT_TRUE(x4Exp.isSameShape(x4));
    ASSERT_TRUE(x4Exp.equalsTo(x4));
//    for(int i = 0; i < shape::shapeInfoLength(x4.rankOf()); ++i)
//        ASSERT_TRUE(x4.shapeInfo()[i] == shapeExpX4[i]);
//    for(int i = 0; i < x4.lengthOf(); ++i)
//        ASSERT_TRUE(x4.e<float>(i) == buffExpX4[i]);

    NDArray x5 = x(3, {2});
    NDArray x5Exp(buffExpX5, shapeExpX5);
    ASSERT_TRUE(x5Exp.isSameShape(x5));
    ASSERT_TRUE(x5Exp.equalsTo(x5));

//    for(int i = 0; i < shape::shapeInfoLength(x5.rankOf()); ++i)
//        ASSERT_TRUE(x5.shapeInfo()[i] == shapeExpX5[i]);
//    for(int i = 0; i < x5.lengthOf(); ++i)
//        ASSERT_TRUE(x5.e<float>(i) == buffExpX5[i]);

    // ******************* //
    NDArray y0 = y(0, {1,2});
    NDArray y0Exp(buffExpY0, shapeExpY0);
    ASSERT_TRUE(y0Exp.isSameShape(y0));
    ASSERT_TRUE(y0Exp.equalsTo(y0));
//    for(int i = 0; i < shape::shapeInfoLength(y0.rankOf()); ++i)
//        ASSERT_TRUE(y0.shapeInfo()[i] == shapeExpY0[i]);
//    for(int i = 0; i < y0.lengthOf(); ++i)
//        ASSERT_TRUE(y0.e<float>(i) == buffExpY0[i]);

    NDArray y1 = y(1, {1,2});
    NDArray y1Exp(buffExpY1, shapeExpY1);
    ASSERT_TRUE(y1Exp.isSameShape(y1));
    ASSERT_TRUE(y1Exp.equalsTo(y1));
//    for(int i = 0; i < shape::shapeInfoLength(y1.rankOf()); ++i)
//        ASSERT_TRUE(y1.shapeInfo()[i] == shapeExpY1[i]);
//    for(int i = 0; i < y1.lengthOf(); ++i)
//        ASSERT_TRUE(y1.e<float>(i) == buffExpY1[i]);

    NDArray y2 = y(0, {1,2}, true);
    NDArray y2Exp(buffExpY2, shapeExpY2);
    ASSERT_TRUE(y2Exp.isSameShape(y2));
    ASSERT_TRUE(y2Exp.equalsTo(y2));
//    for(int i = 0; i < shape::shapeInfoLength(y2.rankOf()); ++i)
//        ASSERT_TRUE(y2.shapeInfo()[i] == shapeExpY2[i]);
//    for(int i = 0; i < y2.lengthOf(); ++i)
//        ASSERT_TRUE(y2.e<float>(i) == buffExpY2[i]);

    NDArray y3 = y(2, {1});
    NDArray y3Exp(buffExpY3, shapeExpY3);
    ASSERT_TRUE(y3Exp.isSameShape(y3));
    ASSERT_TRUE(y3Exp.equalsTo(y3));
//    for(int i = 0; i < shape::shapeInfoLength(y3.rankOf()); ++i)
//        ASSERT_TRUE(y3.shapeInfo()[i] == shapeExpY3[i]);
//    for(int i = 0; i < y3.lengthOf(); ++i)
//        ASSERT_TRUE(y3.e<float>(i) == buffExpY3[i]);

    NDArray y4 = y(2, {1}, true);
    NDArray y4Exp = NDArrayFactory::create<float>('f', {2,1,4}, {5, 6, 11, 12, 17, 18, 23, 24});
    ASSERT_TRUE(y4Exp.isSameShape(y4));
    ASSERT_TRUE(y4Exp.equalsTo(y4));
//    for(int i = 0; i < shape::shapeInfoLength(y4.rankOf()); ++i)
//        ASSERT_TRUE(y4.shapeInfo()[i] == shapeExpY4[i]);
//    for(int i = 0; i < y4.lengthOf(); ++i)
//        ASSERT_TRUE(y4.e<float>(i) == buffExpY4[i]);

    NDArray y5 = y(3, {2});
    NDArray y5Exp(buffExpY5, shapeExpY5);
    ASSERT_TRUE(y5Exp.isSameShape(y5));
    ASSERT_TRUE(y5Exp.equalsTo(y5));
//    for(int i = 0; i < shape::shapeInfoLength(y5.rankOf()); ++i)
//        ASSERT_TRUE(y5.shapeInfo()[i] == shapeExpY5[i]);
//    for(int i = 0; i < y5.lengthOf(); ++i)
//        ASSERT_TRUE(y5.e<float>(i) == buffExpY5[i]);

}
//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayCudaBasicsTests, Test_diagonal_1) {

    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    auto exp = NDArrayFactory::create<float>('c', {2, 1}, {1, 5});

    auto diag = x.diagonal('c');
    //diag.syncToDevice();
    for (Nd4jLong e = 0; e < exp.lengthOf(); ++e) {
        printf("VAL[%ld] = %f\n", e, diag.e<float>(e)); //, exp.e<float>(e), 1.e-5);
    }

    for (Nd4jLong e = 0; e < exp.lengthOf(); ++e) {
        ASSERT_NEAR(diag.e<float>(e), exp.e<float>(e), 1.e-5);
    }
    double eps(1.e-5);
    NDArray tmp(sd::DataType::FLOAT32, x.getContext()); // scalar = 0

    ExtraArguments extras({eps});
    NativeOpExecutioner::execReduce3Scalar(diag.getContext(), reduce3::EqualsWithEps, diag.buffer(),
            diag.shapeInfo(), diag.specialBuffer(), diag.specialShapeInfo(), extras.argumentsAsT(sd::DataType::FLOAT32),
            exp.buffer(), exp.shapeInfo(), exp.specialBuffer(), exp.specialShapeInfo(),
            tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo());
    cudaStream_t* stream = x.getContext()->getCudaStream();
    auto res = cudaStreamSynchronize(*stream);
    // tmp.printBuffer("Compare result is (expected 0)");
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
}

TEST_F(NDArrayCudaBasicsTests, Test_PermuteEquality_02) {
    auto x = NDArrayFactory::linspace<float>(1.f, 60.f, 60); //('c', {1, 60});
    //x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0});
    x->reshapei('c', {3, 4, 5});

    x->permutei({0, 1, 2});
    x->streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(x));
    ASSERT_TRUE(exp.equalsTo(x));
    delete x;
}

TEST_F(NDArrayCudaBasicsTests, Test_PermuteEquality_0) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 1, 2});
    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}
TEST_F(NDArrayCudaBasicsTests, Test_PermuteEquality_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 1, 2});
    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}
TEST_F(NDArrayCudaBasicsTests, Test_PermuteEquality_2) {
    //auto x = NDArrayFactory::create<float>('c', {1, 60});
    auto xx = NDArrayFactory::linspace<float>(1.f, 60.f, 60); //('c', {1, 60});
//    auto x = *xx;
    //x.linspace(1);
//    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0});
//    x.reshapei('c', {3, 4, 5});

//    x.permutei({0, 1, 2});
//    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

//    ASSERT_TRUE(exp.isSameShape(&x));
//    ASSERT_TRUE(exp.equalsTo(&x));
    delete xx;
}
TEST_F(NDArrayCudaBasicsTests, Test_PermuteEquality_3) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    //x.linspace(1);
    for (int l = 0; l < x.lengthOf(); l++)
        x.p(l, float(l + 1.f));
    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 1, 2});
    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayCudaBasicsTests, Test_Empty_1) {
    auto x = NDArrayFactory::empty<float>();
    ASSERT_TRUE(x.isActualOnHostSide());
    ASSERT_TRUE(x.isEmpty());
}

TEST_F(NDArrayCudaBasicsTests, Test_Empty_2) {
    auto x = NDArrayFactory::empty_<float>();

    ASSERT_TRUE(x->isEmpty());
    delete x;
}

TEST_F(NDArrayCudaBasicsTests, Test_Empty_3) {
    auto x = NDArrayFactory::empty(sd::DataType::FLOAT32);

    ASSERT_TRUE(x.isEmpty());
}

TEST_F(NDArrayCudaBasicsTests, Test_Empty_4) {
    auto x = NDArrayFactory::empty_(sd::DataType::FLOAT32);

    ASSERT_TRUE(x->isEmpty());
    delete x;
}