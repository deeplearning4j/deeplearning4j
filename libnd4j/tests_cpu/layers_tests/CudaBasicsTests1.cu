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
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <graph/Context.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/specials_cuda.h>
#include <helpers/TAD.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <cuda.h>
#include <helpers/RandomLauncher.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array/ShapeDescriptor.h>
#include <array/ConstantDataBuffer.h>
#include <helpers/ShapeUtils.h>

using namespace sd;
using namespace sd::graph;

class CudaBasicsTests1 : public testing::Test {
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

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, TestPairwise_1) {
	// allocating host-side arrays
	auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
	auto z = NDArrayFactory::create<double>('c', { 5 }, {0,0,0,0,0});

	auto exp = NDArrayFactory::create<double>('c', { 5 }, { 2, 4, 6, 8, 10 });

	// making raw buffers
	Nd4jPointer devBufferPtrX, devBufferPtrZ, devShapePtrX;
	cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrX), x.lengthOf() * x.sizeOfT());
	ASSERT_EQ(0, res);
	res = cudaMalloc(reinterpret_cast<void **>(&devBufferPtrZ), x.lengthOf() * x.sizeOfT());
	ASSERT_EQ(0, res);
	res = cudaMalloc(reinterpret_cast<void **>(&devShapePtrX), shape::shapeInfoByteLength(x.shapeInfo()));
	ASSERT_EQ(0, res);

	Nd4jPointer nativeStream = (Nd4jPointer)malloc(sizeof(cudaStream_t));
	CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));
	cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
	auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

    x.dataBuffer()->allocatePrimary();
    x.syncToHost();

	cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
	cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);
    res = cudaStreamSynchronize(*stream);
    ASSERT_EQ(0, res);

	LaunchContext lc(stream, nullptr, nullptr);
	NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, nullptr, x.shapeInfo(), devBufferPtrX, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr, x.shapeInfo(), devBufferPtrX, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr, z.shapeInfo(), devBufferPtrZ, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr);
	res = cudaStreamSynchronize(*stream);
	ASSERT_EQ(0, res);

	z.dataBuffer()->allocatePrimary();

	cudaMemcpyAsync(z.buffer(), devBufferPtrZ, z.lengthOf() * x.sizeOfT(), cudaMemcpyDeviceToHost, *stream);
	res = cudaStreamSynchronize(*stream);
	ASSERT_EQ(0, res);

	cudaFree(devBufferPtrX);
	cudaFree(devBufferPtrZ);
	cudaFree(devShapePtrX);

	// needed due to memcpy
    z.tickWriteHost();

	for (int e = 0; e < z.lengthOf(); e++) {
	    //nd4j_printf("step %i\n", e);
		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
	}
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execIndexReduceScalar_1) {

    NDArray x1('c', {2,2}, {0, 1, 2, 3}, sd::DataType::INT32);
    NDArray x2('c', {2,2}, {0.5, 1.5, -4.5, 3.5}, sd::DataType::BFLOAT16);
    NDArray x3('c', {2,2}, {0, -1, 0, 1}, sd::DataType::BOOL);

    NDArray scalar('c', {}, std::vector<double>{0}, sd::DataType::INT64);

    NDArray exp1('c', {}, std::vector<double>{3}, sd::DataType::INT64);
    NDArray exp2('c', {}, std::vector<double>{2}, sd::DataType::INT64);
    NDArray exp3('c', {}, std::vector<double>{1}, sd::DataType::INT64);

    void *dX1, *dX2, *dX3, *dZ;
    Nd4jLong *dX1ShapeInfo, *dX2ShapeInfo, *dX3ShapeInfo, *dZShapeInfo;

    cudaError_t cudaResult;

    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1), x1.lengthOf() * x1.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2), x2.lengthOf() * x2.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3), x3.lengthOf() * x3.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ), scalar.lengthOf() * scalar.sizeOfT()); 				         ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1ShapeInfo), shape::shapeInfoByteLength(x1.shapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2ShapeInfo), shape::shapeInfoByteLength(x2.shapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3ShapeInfo), shape::shapeInfoByteLength(x3.shapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZShapeInfo), shape::shapeInfoByteLength(scalar.shapeInfo())); ASSERT_EQ(0, cudaResult);

    cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);
	ASSERT_EQ(0, cudaResult);

	x1.syncToHost();
	x2.syncToHost();
	x3.syncToHost();
	scalar.syncToHost();

	cudaMemcpyAsync(dX1, x1.buffer(), x1.lengthOf() * x1.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX2, x2.buffer(), x2.lengthOf() * x2.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX3, x3.buffer(), x3.lengthOf() * x3.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX1ShapeInfo, x1.shapeInfo(), shape::shapeInfoByteLength(x1.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX2ShapeInfo, x2.shapeInfo(), shape::shapeInfoByteLength(x2.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX3ShapeInfo, x3.shapeInfo(), shape::shapeInfoByteLength(x3.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZShapeInfo, scalar.shapeInfo(), shape::shapeInfoByteLength(scalar.shapeInfo()), cudaMemcpyHostToDevice, stream);

	void* reductionPointer = nullptr;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024);
	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMemset(reductionPointer, 0, 1024 * 1024);
    ASSERT_EQ(0, cudaResult);

	LaunchContext lc(&stream, LaunchContext::defaultContext()->getReductionPointer(), LaunchContext::defaultContext()->getScalarPointer(), LaunchContext::defaultContext()->getAllocationPointer());

	/***************************************/

    NativeOpExecutioner::execIndexReduceScalar(&lc,
    											sd::indexreduce::IndexAbsoluteMax,
    											x1.buffer(), x1.shapeInfo(),
    	                                       	dX1, dX1ShapeInfo,
    	                                       	nullptr,
    	                                       	scalar.buffer(), scalar.shapeInfo(),
    	                                       	dZ, dZShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    cudaMemcpyAsync(scalar.buffer(), dZ, scalar.lengthOf() * scalar.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    scalar.tickWriteHost();

	ASSERT_NEAR(exp1.e<float>(0), scalar.e<float>(0), 1e-5);

    /***************************************/

    NativeOpExecutioner::execIndexReduceScalar(&lc,
    											sd::indexreduce::IndexAbsoluteMax,
    											nullptr, x2.shapeInfo(),
    	                                       	dX2, dX2ShapeInfo,
    	                                       	nullptr,
    	                                       	nullptr, scalar.shapeInfo(),
    	                                       	dZ, dZShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    cudaMemcpyAsync(scalar.buffer(), dZ, scalar.lengthOf() * scalar.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    ASSERT_NEAR(exp2.e<float>(0), scalar.e<float>(0), 1e-5);

    // *************************************

    NativeOpExecutioner::execIndexReduceScalar(&lc,
    											sd::indexreduce::IndexAbsoluteMax,
    											nullptr, x3.shapeInfo(),
    	                                       	dX3, dX3ShapeInfo,
    	                                       	nullptr,
    	                                       	nullptr, scalar.shapeInfo(),
    	                                       	dZ, dZShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    cudaMemcpyAsync(scalar.buffer(), dZ, scalar.lengthOf() * scalar.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    ASSERT_NEAR(exp3.e<float>(0), scalar.e<float>(0), 1e-5);

	/***************************************/

	cudaFree(dX1); 			cudaFree(dX2); 			cudaFree(dX3); 			cudaFree(dZ);
	cudaFree(dX1ShapeInfo); cudaFree(dX2ShapeInfo); cudaFree(dX3ShapeInfo); cudaFree(dZShapeInfo);

	/***************************************/

	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);

}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3Scalar_1) {

	 if (!Environment::getInstance().isExperimentalBuild())
        return;

    NDArray x1('c', {2,2}, {1,2,3,4}, sd::DataType::INT32);
    NDArray x2('c', {2,2}, {-1,-2,-3,-4}, sd::DataType::INT32);
    NDArray x3('c', {2,2}, {1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);
    NDArray x4('c', {2,2}, {1,2,3,4}, sd::DataType::DOUBLE);

    NDArray exp1('c', {}, std::vector<double>{-30.f}, sd::DataType::FLOAT32);
    NDArray exp2('c', {}, std::vector<double>{15.}, sd::DataType::DOUBLE);

	NDArray scalar1('c', {}, std::vector<double>{100.f}, sd::DataType::FLOAT32);
    NDArray scalar2('c', {}, std::vector<double>{100.}, sd::DataType::DOUBLE);

    void *dX1, *dX2, *dX3, *dX4, *dZ1, *dZ2;
    Nd4jLong *dX1ShapeInfo, *dX3ShapeInfo, *dZ1ShapeInfo, *dZ2ShapeInfo;

    cudaError_t cudaResult;

    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1), x1.lengthOf() * x1.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2), x2.lengthOf() * x2.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3), x3.lengthOf() * x3.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX4), x4.lengthOf() * x4.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1), scalar1.lengthOf() * scalar1.sizeOfT());			         	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2), scalar2.lengthOf() * scalar2.sizeOfT());			         	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1ShapeInfo), shape::shapeInfoByteLength(x1.shapeInfo()));    	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3ShapeInfo), shape::shapeInfoByteLength(x3.shapeInfo()));    	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1ShapeInfo), shape::shapeInfoByteLength(scalar1.shapeInfo())); 	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2ShapeInfo), shape::shapeInfoByteLength(scalar2.shapeInfo())); 	ASSERT_EQ(0, cudaResult);

    cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);
	ASSERT_EQ(0, cudaResult);

	x1.syncToHost();
	x2.syncToHost();
	x3.syncToHost();
	x4.syncToHost();
	scalar1.syncToHost();
	scalar2.syncToHost();

	cudaMemcpyAsync(dX1, x1.buffer(), x1.lengthOf() * x1.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX2, x2.buffer(), x2.lengthOf() * x2.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX3, x3.buffer(), x3.lengthOf() * x3.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX4, x4.buffer(), x4.lengthOf() * x4.sizeOfT(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX1ShapeInfo, x1.shapeInfo(), shape::shapeInfoByteLength(x1.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX3ShapeInfo, x3.shapeInfo(), shape::shapeInfoByteLength(x3.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZ1ShapeInfo, scalar1.shapeInfo(), shape::shapeInfoByteLength(scalar1.shapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZ2ShapeInfo, scalar2.shapeInfo(), shape::shapeInfoByteLength(scalar2.shapeInfo()), cudaMemcpyHostToDevice, stream);

	/***************************************/

	void* reductionPointer  = nullptr;
	int*  allocationPointer = nullptr;

	cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024);		ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024);		ASSERT_EQ(0, cudaResult);

	LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);

	/***************************************/

    NativeOpExecutioner::execReduce3Scalar(&lc, sd::reduce3::Dot,nullptr, x1.shapeInfo(),dX1, dX1ShapeInfo, nullptr, nullptr, x2.shapeInfo(),dX2, dX1ShapeInfo,nullptr, scalar1.shapeInfo(),dZ1, dZ1ShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    scalar1.tickWriteHost();
    scalar2.tickWriteHost();

    cudaMemcpyAsync(scalar1.buffer(), dZ1, scalar1.lengthOf() * scalar1.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

	ASSERT_NEAR(exp1.e<float>(0), scalar1.e<float>(0), 1e-5);

    /***************************************/

    NativeOpExecutioner::execReduce3Scalar(&lc, sd::reduce3::Dot,nullptr, x3.shapeInfo(),dX3, dX3ShapeInfo, nullptr, nullptr, x4.shapeInfo(),dX4, dX3ShapeInfo,nullptr, scalar2.shapeInfo(),dZ2, dZ2ShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

    cudaMemcpyAsync(scalar2.buffer(), dZ2, scalar2.lengthOf() * scalar2.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);

	ASSERT_NEAR(exp2.e<float>(0), scalar2.e<float>(0), 1e-5);

	/***************************************/

	cudaFree(dX1); 			cudaFree(dX2); cudaFree(dX3); 		   cudaFree(dX4); 	cudaFree(dZ1); 				cudaFree(dZ2);
	cudaFree(dX1ShapeInfo); 			   cudaFree(dX3ShapeInfo); 					cudaFree(dZ1ShapeInfo);		cudaFree(dZ2ShapeInfo);

	/***************************************/

	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3_1) {

    NDArray x('c', {2,2}, {1,2,3,4}, sd::DataType::INT32);
    NDArray y('c', {2,2}, {-1,-2,-3,-4}, sd::DataType::INT32);

    NDArray exp('c', {}, std::vector<double>{-30.f}, sd::DataType::FLOAT32);
    NDArray z('c', {}, std::vector<double>{100.f},  sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0, 1};

    x.syncToHost();
    y.syncToHost();
    z.syncToHost();


    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

    cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								nullptr, nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3_2) {

	NDArray x('c', {2,2}, {1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);
    NDArray y('c', {2,2}, {1,2,3,4}, sd::DataType::DOUBLE);

    NDArray exp('c', {}, std::vector<double>{15.}, sd::DataType::DOUBLE);
    NDArray z('c', {}, std::vector<double>{100.},  sd::DataType::DOUBLE);

    std::vector<int> dimensions = {0, 1};

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								nullptr, nullptr, nullptr, nullptr);


	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3_3) {

	NDArray x('c', {2,3}, {1,2,3,4,5,6}, sd::DataType::INT32);
    NDArray y('c', {2,3}, {-6,-5,-4,-3,-2,-1}, sd::DataType::INT32);

    NDArray exp('c', {3}, {-18,-20,-18}, sd::DataType::FLOAT32);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.shapeInfo(), dimensions.data(), dimensions.size());
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	hostData.emplace_back(yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));// 3 -- yTadShapeInfo
	hostData.emplace_back(yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong));						// 4-- yTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
								(Nd4jLong*)devicePtrs[3], (Nd4jLong*)devicePtrs[4]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
	z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3_4) {

    NDArray x('c', {2,3}, {1,2,3,4,5,6}, sd::DataType::DOUBLE);
    NDArray y('c', {2,3}, {1.5,1.5,1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);

    NDArray exp('c', {2}, {9,22.5}, sd::DataType::DOUBLE);
    NDArray z('c', {2}, {100,100}, sd::DataType::DOUBLE);

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.shapeInfo(), dimensions.data(), dimensions.size());
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	hostData.emplace_back(yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));// 3 -- yTadShapeInfo
	hostData.emplace_back(yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong));						// 4-- yTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
								(Nd4jLong*)devicePtrs[3], (Nd4jLong*)devicePtrs[4]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3_5) {

    NDArray x('c', {2,2,3}, {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5}, sd::DataType::FLOAT32);
    NDArray y('c', {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::FLOAT32);

    NDArray exp('c', {2,3}, {7.5, 10.5, 13.5, 25.5, 28.5, 31.5}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.shapeInfo(), dimensions.data(), dimensions.size());
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	hostData.emplace_back(yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));// 3 -- yTadShapeInfo
	hostData.emplace_back(yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong));						// 4-- yTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
								(Nd4jLong*)devicePtrs[3], (Nd4jLong*)devicePtrs[4]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3All_1) {

    NDArray x('c', {2,2}, {1,2,3,4}, sd::DataType::INT32);
    NDArray y('c', {2,3}, {-1,1,-1,1,-1,1}, sd::DataType::INT32);

    NDArray exp('c', {2,3}, {2,-2,2,2,-2,2}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.shapeInfo(), dimensions.data(), dimensions.size());
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	hostData.emplace_back(yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));// 3 -- yTadShapeInfo
	hostData.emplace_back(yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong));						// 4 -- yTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3All(&lc, sd::reduce3::Dot,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr,
										nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										(int*)devicePtrs[0], dimensions.size(),
										(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
										(Nd4jLong*)devicePtrs[3], (Nd4jLong*)devicePtrs[4]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
	z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3All_2) {

    NDArray x('c', {2,2}, {1,2,3,4}, sd::DataType::DOUBLE);
    NDArray y('c', {2,3}, {1.5,1.5,1.5,1.5,1.5,1.5}, sd::DataType::DOUBLE);

    NDArray exp('c', {2,3}, {6,6,6,9,9,9}, sd::DataType::DOUBLE);
    NDArray z('c', {2,3}, {100,100,100,100,100,100,},sd::DataType::DOUBLE);

    std::vector<int> dimensions = {0};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.shapeInfo(), dimensions.data(), dimensions.size());
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	hostData.emplace_back(yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));// 3 -- yTadShapeInfo
	hostData.emplace_back(yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong));						// 4-- yTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3All(&lc, sd::reduce3::Dot,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr,
										nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										(int*)devicePtrs[0], dimensions.size(),
										(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
										(Nd4jLong*)devicePtrs[3], (Nd4jLong*)devicePtrs[4]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execIndexReduce_1) {

    NDArray x('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::DOUBLE);
    x.linspace(-2.); x.syncToDevice();
    NDArray exp('c', {2}, {2, 2}, sd::DataType::INT64);
    NDArray z('c', {2}, {100,100}, sd::DataType::INT64);

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execIndexReduce(&lc, sd::indexreduce::IndexMax,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr,
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										(int*)devicePtrs[0], dimensions.size(),
										(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2]);

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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execIndexReduce_2) {

    NDArray x('c', {2,3,4,5}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    						  	100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::FLOAT32);
    x.linspace(-2.f); x.syncToDevice();
    NDArray exp('c', {2,5}, {11,11,11,11,11,11,11,11,11,11}, sd::DataType::INT64);
    NDArray z('c', {2,5}, {100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT64);

    std::vector<int> dimensions = {1,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function

    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execIndexReduce(&lc, sd::indexreduce::IndexMax,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr,
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										(int*)devicePtrs[0], dimensions.size(),
										(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2]);

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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execIndexReduce_3) {

    NDArray x('c', {2,3,4,5}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    						  	100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
    x.linspace(-2.); x.syncToDevice();
    NDArray exp('c', {3}, {39, 39, 39}, sd::DataType::INT64);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::INT64);

    std::vector<int> dimensions = {0,2,3};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execIndexReduce(&lc, sd::indexreduce::IndexMax,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr,
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										(int*)devicePtrs[0], dimensions.size(),
										(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2]);

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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execScalar_1) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

    NDArray x('c', {2,3},  {0,1,2,3,4,5}, sd::DataType::INT64);
    NDArray exp('c',{2,3}, {0,0,1,1,2,2}, sd::DataType::INT64);
    NDArray scalar('c',{}, std::vector<double>{2.f}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::INT64);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalar(&lc, sd::scalar::Divide,
									nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
									nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(),
									nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execScalar_2) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

    NDArray x('c', {2,3},  {-1,-2,-3,-4,-5,-6}, sd::DataType::INT64);
    NDArray exp('c',{2,3}, {10,10,10,10,10,10}, sd::DataType::FLOAT32);
    NDArray scalar('c',{}, std::vector<double>{10.f}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalar(&lc, sd::scalar::CopyPws,
									nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
									nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(),
									nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);


	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execScalar_3) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

    NDArray x('c', {2,3,2},  {0,1,2,3,4,5,6,7,8,9,10,11}, sd::DataType::INT64);
    NDArray scalars('c',{2,2}, {1,2,3,4}, sd::DataType::FLOAT32);
    NDArray exp('c', {2,3,2},  {0,0,2,1,4,2, 2,1,2,2,3,2}, sd::DataType::INT64);
    NDArray z('c', {2,3,2}, {100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT64);

    std::vector<int> dimensions = {1};

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
	NativeOpExecutioner::execScalar(&lc, sd::scalar::Divide,
									nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
									nullptr,
									nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalars.shapeInfo(), scalars.specialBuffer(), scalars.specialShapeInfo(),
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execScalarBool_1) {

    NDArray x('c', {2,3},  {-1,-2,0,1,2,3}, sd::DataType::BFLOAT16);
    NDArray scalar('c',{}, std::vector<double>{0}, sd::DataType::BFLOAT16);
    NDArray exp('c',{2,3}, {0,0,0,1,1,1}, sd::DataType::BOOL);
    NDArray z('c', {2,3}, {100,100,100,100,100,100,}, sd::DataType::BOOL);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalarBool(&lc, sd::scalar::GreaterThan,
									nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
									nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalar.shapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(),
									nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execScalarBool_2) {

    NDArray x('c', {2,3},  {0,1,2,3,4,5}, sd::DataType::FLOAT32);
    NDArray scalars('c',{2}, {-1,4}, sd::DataType::FLOAT32);
    NDArray exp('c', {2,3},  {1,1,1,0,0,1}, sd::DataType::BOOL);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, sd::DataType::BOOL);

    std::vector<int> dimensions = {1};

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
	NativeOpExecutioner::execScalarBool(&lc, sd::scalar::GreaterThan,
									nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
									nullptr,
									nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalars.shapeInfo(), scalars.specialBuffer(), scalars.specialShapeInfo(),
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execBroadcast_1) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
    NDArray y('c', {3},   {10, 20, 30}, sd::DataType::INT64);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
	NDArray exp('c', {2,3,4}, {10, 11, 12, 13,24, 25, 26, 27,38, 39, 40, 41,22, 23, 24, 25,36, 37, 38, 39,50, 51, 52, 53}, sd::DataType::INT32);
	x.linspace(0); x.syncToDevice();

    std::vector<int> dimensions = {1};

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
	NativeOpExecutioner::execBroadcast(&lc, sd::broadcast::Add,
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execBroadcast_2) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::FLOAT32);
	NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, sd::DataType::FLOAT32);
	x.linspace(0); x.syncToDevice();

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
	NativeOpExecutioner::execBroadcast(&lc, sd::broadcast::Add,
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execBroadcastBool_1) {

	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::INT32);
    NDArray y('c', {3},   {2, 12, 22}, sd::DataType::INT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,}, sd::DataType::BOOL);
	NDArray exp('c', {2,3,4}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, sd::DataType::BOOL);
	x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {1};

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
	NativeOpExecutioner::execBroadcastBool(&lc, sd::broadcast::EqualTo,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                        nullptr,
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execBroadcastBool_2) {

	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100},sd::DataType::FLOAT32);
    NDArray y('c', {2,4},   {1,10,10,15,20,20,20,24}, sd::DataType::FLOAT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, sd::DataType::BOOL);
	NDArray exp('c', {2,3,4}, {1, 0, 0, 0,0, 0, 0, 0,0, 1, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 1}, sd::DataType::BOOL);
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
	NativeOpExecutioner::execBroadcastBool(&lc, sd::broadcast::EqualTo,
										nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
										nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
										nullptr,
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execPairwiseTransform_1) {

	if (!Environment::getInstance().isExperimentalBuild())
        return;

	NDArray x('c', {2,2,2}, {1,5,3,7,2,6,4,8}, sd::DataType::INT32);
    NDArray y('c', {4,2}, {0.1,0.2,0.3,0.4,1.5,0.6,0.7,1.8}, sd::DataType::DOUBLE);
    NDArray z('c', {8}, {100,100,100,100,100,100,100,100}, sd::DataType::INT32);
	NDArray exp('c', {8}, {0,1,2,3,3,5,6,6}, sd::DataType::INT32);
	x.permutei({2,1,0});	// -> {1,2,3,4,5,6,7,8}
    x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execPairwiseTransform(&lc, sd::pairwise::Subtract,
												nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
												nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
												nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
												nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execPairwiseBoolTransform_1) {

	NDArray x('c', {2,2,2}, {1,5,3,7,2,6,4,8}, sd::DataType::INT64);
    NDArray y('c', {4,2}, {0,2,0,4,0,6,0,8}, sd::DataType::INT64);
    NDArray z('c', {8}, {100,100,100,100,100,100,100,100}, sd::DataType::BOOL);
	NDArray exp('c', {8}, {0,1,0,1,0,1,0,1}, sd::DataType::BOOL);
	x.permutei({2,1,0});	// -> {1,2,3,4,5,6,7,8}
	x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execPairwiseBoolTransform(&lc, sd::pairwise::EqualTo,
													nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
													nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
													nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
													nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformFloat_1) {

	NDArray x('c', {2,2}, {0, 6.25, 2.25, 12.25}, sd::DataType::DOUBLE);
    NDArray z('c', {4}, {100,100,100,100}, sd::DataType::FLOAT32);
	NDArray exp('c', {4}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
	x.permutei({1,0});
	x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformFloat(&lc, sd::transform::Sqrt,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformFloat_2) {

	NDArray x('c', {1,4}, {0, 4, 9, 16}, sd::DataType::INT64);
    NDArray z('c', {2,2}, {100,100,100,100}, sd::DataType::DOUBLE);
	NDArray exp('c', {2,2}, {0, 2, 3, 4}, sd::DataType::DOUBLE);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformFloat(&lc, sd::transform::Sqrt,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformAny_1) {

	NDArray x('c', {2,2}, {0, 6.25, 2.25, 12.25}, sd::DataType::DOUBLE);
    NDArray z('c', {4,1}, {100,100,100,100}, sd::DataType::INT32);
	NDArray exp('c', {4,1}, {0, 2, 6, 12}, sd::DataType::INT32);
	x.permutei({1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformAny(&lc, sd::transform::Assign,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformAny_2) {

	NDArray x('c', {1,4}, {0, 6.25, 2.25, 12.25}, sd::DataType::BFLOAT16);
    NDArray z('c', {2,2}, {100,100,100,100}, sd::DataType::FLOAT32);
	NDArray exp('c', {2,2}, {0, 6.25, 2.25, 12.25}, sd::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformAny(&lc, sd::transform::Assign,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformStrict_1) {

	NDArray x('c', {2,3}, {0,2,4,1,3,5}, sd::DataType::DOUBLE);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::DOUBLE);
	NDArray exp('c', {3,2}, {0, 3, 12, 27, 48, 75}, sd::DataType::DOUBLE);
	x.permutei({1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformStrict(&lc, sd::transform::CubeDerivative,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformStrict_2) {

	NDArray x('c', {6}, {0,1,2,3,4,5}, sd::DataType::FLOAT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::FLOAT32);
	NDArray exp('c', {3,2}, {0, 3, 12, 27, 48, 75}, sd::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformStrict(&lc, sd::transform::CubeDerivative,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
	z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformSame_1) {

	NDArray x('c', {2,3}, {0,2.5,4.5,1.5,3.5,5.5}, sd::DataType::DOUBLE);
    NDArray z('c', {1,6}, {100,100,100,100,100,100}, sd::DataType::DOUBLE);
	NDArray exp('c', {1,6}, {0,2.25,6.25,12.25,20.25,30.25}, sd::DataType::DOUBLE);
	x.permutei({1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformSame(&lc, sd::transform::Square,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformSame_2) {

	NDArray x('c', {6}, {0,1,2,3,4,5}, sd::DataType::INT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::INT32);
	NDArray exp('c', {3,2}, {0,1,4,9,16,25}, sd::DataType::INT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformSame(&lc, sd::transform::Square,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformBool_1) {

	NDArray x('c', {2,3}, {0,2,4,-1,-3,-5}, sd::DataType::DOUBLE);
    NDArray z('c', {1,6}, {100,100,100,100,100,100}, sd::DataType::BOOL);
	NDArray exp('c', {1,6}, {0,0,1,0,1,0}, sd::DataType::BOOL);
	x.permutei({1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformBool(&lc, sd::transform::IsPositive,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execTransformBool_2) {

	NDArray x('c', {6}, {0,-1,2,-3,4,-5}, sd::DataType::INT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, sd::DataType::BOOL);
	NDArray exp('c', {3,2}, {0,0,1,0,1,0}, sd::DataType::BOOL);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformBool(&lc, sd::transform::IsPositive,
		nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
		nullptr, nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceFloat_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::FLOAT32);
    NDArray exp('c', {3}, {2.5, 6.5, 10.5}, sd::DataType::FLOAT32);
    x.permutei({2,1,0});

    std::vector<int> dimensions = {0,2};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceFloat(&lc, sd::reduce::Mean, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceFloat_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
    NDArray exp('c', {2,4}, {-1., 0., 1., 2.,11., 12., 13., 14.}, sd::DataType::DOUBLE);

    std::vector<int> dimensions = {1};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceFloat(&lc, sd::reduce::Mean, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceSame_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::INT32);
    NDArray exp('c', {3}, {20, 52, 84}, sd::DataType::INT32);
    x.permutei({2,1,0});

    std::vector<int> dimensions = {0,2};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceSame(&lc, sd::reduce::Sum, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceSame_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, sd::DataType::FLOAT32);
    NDArray exp('c', {2,4}, {-3., 0., 3., 6.,33., 36., 39., 42.}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {1};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceSame(&lc, sd::reduce::Sum, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceBool_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, sd::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::BOOL);
    NDArray exp('c', {3}, {0, 1, 1}, sd::DataType::BOOL);
    x.permutei({2,1,0});

    std::vector<int> dimensions = {0,2};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceBool(&lc, sd::reduce::IsPositive, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceBool_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, sd::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, sd::DataType::BOOL);
    NDArray exp('c', {2,4}, {1, 1, 1, 1, 0, 0, 0, 0}, sd::DataType::BOOL);

    std::vector<int> dimensions = {1};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceBool(&lc, sd::reduce::IsPositive, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceLong_1) {

    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, sd::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::INT64);
    NDArray exp('c', {3}, {5,6,6}, sd::DataType::INT64);
    x.permutei({2,1,0});

    std::vector<int> dimensions = {0,2};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceLong(&lc, sd::reduce::CountNonZero, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceLong_2) {

    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, sd::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, sd::DataType::INT64);
    NDArray exp('c', {2,4}, {3, 1, 3, 2, 2, 1, 2, 3}, sd::DataType::INT64);

    std::vector<int> dimensions = {1};

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	std::vector<int> dims = sd::ShapeUtils::evalDimsForReduceOp(x.rankOf(), dimensions);
    NativeOpExecutioner::execReduceLong(&lc, sd::reduce::CountNonZero, nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dims.data(), dims.size());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream);
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceFloatScalar_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::FLOAT32);
    NDArray exp('c', {}, std::vector<double>{6.5}, sd::DataType::FLOAT32);
    x.permutei({2,1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceFloatScalar(&lc, sd::reduce::Mean,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceFloatScalar_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::DOUBLE);
    NDArray exp('c', {}, std::vector<double>{6.5}, sd::DataType::DOUBLE);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceFloatScalar(&lc, sd::reduce::Mean,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceSameScalar_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::INT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::INT32);
    NDArray exp('c', {}, std::vector<double>{156}, sd::DataType::INT32);
    x.permutei({2,1,0});

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceSameScalar(&lc, sd::reduce::Sum,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceSameScalar_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, sd::DataType::DOUBLE);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::DOUBLE);
    NDArray exp('c', {}, std::vector<double>{156}, sd::DataType::DOUBLE);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceSameScalar(&lc, sd::reduce::Sum,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceBoolScalar_1) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, sd::DataType::INT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::BOOL);
    NDArray exp('c', {}, std::vector<double>{1}, sd::DataType::BOOL);
    x.permutei({2,1,0});
    x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceBoolScalar(&lc, sd::reduce::IsPositive,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceBoolScalar_2) {

    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, sd::DataType::DOUBLE);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::BOOL);
    NDArray exp('c', {}, std::vector<double>{1}, sd::DataType::BOOL);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceBoolScalar(&lc, sd::reduce::IsPositive,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceLongScalar_1) {

    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, sd::DataType::INT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::INT64);
    NDArray exp('c', {}, std::vector<double>{17}, sd::DataType::INT64);
    x.permutei({2,1,0});
    x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceLongScalar(&lc, sd::reduce::CountNonZero,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduceLongScalar_2) {

    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, sd::DataType::DOUBLE);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::INT64);
    NDArray exp('c', {}, std::vector<double>{17}, sd::DataType::INT64);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024); ASSERT_EQ(0, cudaResult);
    int* allocationPointer;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);
	lc.setAllocationPointer(allocationPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduceLongScalar(&lc, sd::reduce::CountNonZero,
					nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr,
					nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3TAD_1) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::FLOAT32);
    NDArray y('c', {2,2}, {1,2,3,4}, sd::DataType::FLOAT32);
    NDArray exp('c', {3}, {10,20,30}, sd::DataType::DOUBLE);
    NDArray z('c', {3}, {100,100,100}, sd::DataType::DOUBLE);

    std::vector<int> dimensions = {0,1};
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), dimensions);
    LaunchContext* context = x.getContext();

	x.syncToDevice();
	y.syncToDevice();
	PointersManager pm(context, "execReduce3TAD_1");
	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3TAD(context, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								nullptr, dimensions.size(),
								packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
    pm.synchronize();
//	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();
//    z.printIndexedBuffer("OutputReduce3TAD");
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3TAD_2) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::INT64);
    NDArray y('c', {2,3}, {1,2,3,4,5,6}, sd::DataType::INT64);
    NDArray exp('c', {2}, {10,73}, sd::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3TAD(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2], nullptr, nullptr);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3TAD_3) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::INT64);
    NDArray y('c', {3}, {1,2,3}, sd::DataType::INT64);
    NDArray exp('c', {2,2}, {-22,-4,14,32}, sd::DataType::FLOAT32);
    NDArray z('c', {2,2}, {100,100,100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it

	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3TAD(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2], (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execReduce3TAD_4) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::DOUBLE);
    NDArray y('c', {2,2,3}, {10,20,30,40,50,60,70,80,90,100,110,120}, sd::DataType::DOUBLE);
    NDArray exp('c', {}, std::vector<double>{1820}, sd::DataType::FLOAT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0,1,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3TAD(&lc, sd::reduce3::Dot,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2], (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStats_1) {
  // FIXME: Yurii, this test should be fixed
    if (1 > 0)
      return;
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::INT64);
    NDArray exp('c', {}, std::vector<double>{3.605551}, sd::DataType::FLOAT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStats(&lc, sd::variance::SummaryStatsStandardDeviation,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								true);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStats_2) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-20,-1,0,1,2,3,4,5,6}, sd::DataType::DOUBLE);
    NDArray exp('c', {2}, {3.405877, 9.715966}, sd::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStats(&lc, sd::variance::SummaryStatsStandardDeviation,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								(int*)devicePtrs[0], dimensions.size(),
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
								true);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}
/*
////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStats_3) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-20,-1,0,1,2,3,4,5,6}, sd::DataType::DOUBLE);
    NDArray exp('c', {2}, {10.606602, 2.121320}, sd::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, sd::DataType::FLOAT32);

    std::vector<int> dimensions = {1};

    // evaluate xTad data
    shape::TAD xTad;
    xTad.init(x.shapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(dimensions.data(), dimensions.size() * sizeof(int));						// 0 -- dimensions
	hostData.emplace_back(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));// 1 -- xTadShapeInfo
	hostData.emplace_back(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));						// 2 -- xTadOffsets
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStats(&lc, sd::variance::SummaryStatsStandardDeviation,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr, 								
								nullptr, z.shapeInfo(), z.specialBuffer(), z.special(),
								(int*)devicePtrs[0], dimensions.size(), 
								(Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2],
								true);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}
*/

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStatsScalar_1) {

    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, sd::DataType::INT64);
    NDArray exp('c', {}, std::vector<double>{3.605551}, sd::DataType::FLOAT32);
    NDArray z('c', {}, std::vector<double>{100}, sd::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024); ASSERT_EQ(0, cudaResult);
	lc.setReductionPointer(reductionPointer);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStatsScalar(&lc, sd::variance::SummaryStatsStandardDeviation,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr,
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								true);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execRandom_1) {

//    NDArray z('c', {10}, {100,0,0,0,0,0,0,0,0,0}, sd::DataType::DOUBLE);
    NDArray z('c', {10}, {100,0,0,0,0,0,0,0,0,100}, sd::DataType::FLOAT32);
    NDArray exp('c', {10}, {0.050942, -0.183229, -0.093921, 0.075469, 0.257166, -0.254838, 0.342227, -0.682188, -0.004345, 0.464633}, sd::DataType::FLOAT32);

    sd::graph::RandomGenerator gen(119,5);

	cudaError_t cudaResult;
    NDArray* array = &z;
    ExtraArguments arguments({0.f, 0.5f});
    auto context = z.getContext();
    PointersManager pm(context, "tests::execRandom_1");
//    z.printIndexedBuffer("Input data");
//    z.syncToDevice();
    NativeOpExecutioner::execRandom(context, random::GaussianDistribution, &gen, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    pm.synchronize();
    z.tickWriteDevice();
//	z.printIndexedBuffer("Output Gaussian");
//    RandomLauncher::fillGaussian(context, gen, &z,  0.f, 0.5f);
//    pm.synchronize();
//    z.tickWriteDevice();
//    z.printIndexedBuffer("Output Gaussian");

//    cudaStream_t stream;
//    cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
//    LaunchContext lc(&stream);
//
//	//	::execRandom(extraPointers, random::GaussianDistribution, &gen, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.special(), &extra);
//	// call cuda kernel which calculates result
//	NativeOpExecutioner::execRandom(&lc, sd::random::GaussianDistribution,
//								&gen,
//								nullptr, z.shapeInfo(), z.specialBuffer(), z.special(),
//								nullptr, z.shapeInfo(), z.specialBuffer(), z.special(),
//								nullptr, z.shapeInfo(), z.specialBuffer(), z.special(),
//								extraArguments.argumentsAsT(z.dataType()));
//
//	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
//	ASSERT_EQ(cudaResult, 0);
//    z.tickWriteDevice();
//    z.syncToHost();
//    z.printIndexedBuffer("Random1");
    ASSERT_EQ(exp, z);
// 	// verify results
// 	for (int e = 0; e < z.lengthOf(); e++)
// 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
//    cudaFree(dExtraArgs);
	// free allocated global device memory
//	cudaFree(dGen);
	// delete cuda stream
//	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execRandom_2) {

    NDArray x('c', {10}, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}, sd::DataType::DOUBLE);
    NDArray z('c', {2,5}, {100,100,100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
    NDArray exp('c', {10}, {0., 0., 0.3, 0., 0.5, 0., 0.7, 0., 0., 1.}, sd::DataType::DOUBLE);

    ExtraArguments extraArguments({0.7});
    sd::graph::RandomGenerator gen(119,5);

//    // prepare input arrays for prepareDataForCuda function
//    std::vector<std::pair<void*,size_t>> hostData;
//	hostData.emplace_back(extraArguments.data(), extraArguments.size() * sizeof(double));		// 0 -- dimensions
//	std::vector<void*> devicePtrs(hostData.size(), nullptr);
//
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
//	cudaStream_t stream;
//	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext* lc = x.getContext(); //(&stream);

	// allocate required amount of global device memory and copy host data to it
//	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execRandom(lc, sd::random::DropOut,
								&gen,
								nullptr, x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								extraArguments.argumentsAsT(z.dataType()));

	cudaResult = cudaStreamSynchronize(*lc->getCudaStream()); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();
    z.syncToHost();
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
//	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
//	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execRandom_3) {

    NDArray z('c', {10}, {100,100,100,100,100,100,100,100,100,100}, sd::DataType::DOUBLE);
    NDArray exp('c', {10}, {2.373649, 2.239791, 1.887353, 2.488636, 2.068904, 2.281399, 1.828228, 2.228222, 2.490847, 1.669537}, sd::DataType::DOUBLE);

    std::vector<double> extraArguments = {1.5, 2.5};
    sd::graph::RandomGenerator gen(119,5);

    // prepare input arrays for prepareDataForCuda function
    std::vector<std::pair<void*,size_t>> hostData;
	hostData.emplace_back(extraArguments.data(), extraArguments.size() * sizeof(double));		// 0 -- dimensions
	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// allocate required amount of global device memory and copy host data to it
	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execRandom(&lc, sd::random::UniformDistribution,
								&gen,
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								devicePtrs[0]);

	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();

 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execRandom_4) {

    NDArray z('c', {2,5}, {1,2,3,4,5,6,7,8,9,10}, sd::DataType::FLOAT32);
    NDArray exp('c', {10}, {2.373649, 2.281399, 2.239791, 1.828228, 1.887353, 2.228222, 2.488636, 2.490847, 2.068904, 1.669537}, sd::DataType::FLOAT32);
    z.permutei({1,0});

    ExtraArguments extraArguments({1.5, 2.5});
    sd::graph::RandomGenerator gen(119,5);

//    // prepare input arrays for prepareDataForCuda function
//    std::vector<std::pair<void*,size_t>> hostData;
//	hostData.emplace_back(extraArguments.data(), extraArguments.size() * sizeof(double));		// 0 -- dimensions
//	std::vector<void*> devicePtrs(hostData.size(), nullptr);

	// create cuda stream and LaunchContext
//	cudaError_t cudaResult;
//	cudaStream_t stream;
//	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
//	LaunchContext lc(&stream);
//
//	// allocate required amount of global device memory and copy host data to it
//	cudaResult = allocateDeviceMem(lc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
    auto context = z.getContext();
    PointersManager pm(context, "execRandom4");
	// call cuda kernel which calculates result
	NativeOpExecutioner::execRandom(context, sd::random::UniformDistribution,
								&gen,
								nullptr, z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
								extraArguments.argumentsAsT(z.dataType()));

//	cudaResult = cudaStreamSynchronize(stream); ASSERT_EQ(0, cudaResult);
    z.tickWriteDevice();
//    z.printIndexedBuffer("Output Uniform4");
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++)
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	// free allocated global device memory
//	for(int i = 0; i < devicePtrs.size(); ++i) cudaFree(devicePtrs[i]);

	// delete cuda stream
//	cudaResult = cudaStreamDestroy(stream); ASSERT_EQ(0, cudaResult);
}

