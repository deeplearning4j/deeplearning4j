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
#include <specials_cuda.h>
#include <TAD.h>
#include <MmulHelper.h>
#include <helpers/PointersManager.h>
#include <cuda.h>
#include <helpers/RandomLauncher.h>
#include <ConstantShapeHelper.h>
#include <ConstantTadHelper.h>
#include <ShapeDescriptor.h>
#include <array/ConstantDataBuffer.h>

using namespace nd4j;
using namespace nd4j::graph;

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
	    nd4j_printf("step %i\n", e);
		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
	}
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execIndexReduceScalar_1) {

    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray x2('c', {2,2}, {0.5, 1.5, -4.5, 3.5}, nd4j::DataType::BFLOAT16);    
    NDArray x3('c', {2,2}, {0, -1, 0, 1}, nd4j::DataType::BOOL);
    
    NDArray scalar('c', {}, {0}, nd4j::DataType::INT64);

    NDArray exp1('c', {}, {3}, nd4j::DataType::INT64);
    NDArray exp2('c', {}, {2}, nd4j::DataType::INT64);
    NDArray exp3('c', {}, {1}, nd4j::DataType::INT64);

    void *dX1, *dX2, *dX3, *dZ; 
    Nd4jLong *dX1ShapeInfo, *dX2ShapeInfo, *dX3ShapeInfo, *dZShapeInfo;

    cudaError_t cudaResult;

    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1), x1.lengthOf() * x1.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2), x2.lengthOf() * x2.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);    
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3), x3.lengthOf() * x3.sizeOfT()); 		   		         	 ASSERT_EQ(0, cudaResult);    
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ), scalar.lengthOf() * scalar.sizeOfT()); 				         ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1ShapeInfo), shape::shapeInfoByteLength(x1.getShapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2ShapeInfo), shape::shapeInfoByteLength(x2.getShapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3ShapeInfo), shape::shapeInfoByteLength(x3.getShapeInfo()));    ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZShapeInfo), shape::shapeInfoByteLength(scalar.getShapeInfo())); ASSERT_EQ(0, cudaResult);	

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
	cudaMemcpyAsync(dX1ShapeInfo, x1.getShapeInfo(), shape::shapeInfoByteLength(x1.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX2ShapeInfo, x2.getShapeInfo(), shape::shapeInfoByteLength(x2.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dX3ShapeInfo, x3.getShapeInfo(), shape::shapeInfoByteLength(x3.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZShapeInfo, scalar.getShapeInfo(), shape::shapeInfoByteLength(scalar.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	
	void* reductionPointer = nullptr;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024);
	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMemset(reductionPointer, 0, 1024 * 1024);
    ASSERT_EQ(0, cudaResult);

	LaunchContext lc(&stream, LaunchContext::defaultContext()->getReductionPointer(), LaunchContext::defaultContext()->getScalarPointer(), LaunchContext::defaultContext()->getAllocationPointer());

	/***************************************/
	
    NativeOpExecutioner::execIndexReduceScalar(&lc,
    											nd4j::indexreduce::IndexAbsoluteMax, 
    											x1.buffer(), x1.getShapeInfo(),
    	                                       	dX1, dX1ShapeInfo, 
    	                                       	nullptr, 
    	                                       	scalar.buffer(), scalar.getShapeInfo(),
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
    											nd4j::indexreduce::IndexAbsoluteMax, 
    											nullptr, x2.getShapeInfo(),
    	                                       	dX2, dX2ShapeInfo, 
    	                                       	nullptr, 
    	                                       	nullptr, scalar.getShapeInfo(),
    	                                       	dZ, dZShapeInfo);

    cudaResult = cudaStreamSynchronize(stream); 
    ASSERT_EQ(0, cudaResult);

    cudaMemcpyAsync(scalar.buffer(), dZ, scalar.lengthOf() * scalar.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream); 
    ASSERT_EQ(0, cudaResult);

    ASSERT_NEAR(exp2.e<float>(0), scalar.e<float>(0), 1e-5);

    // *************************************

    NativeOpExecutioner::execIndexReduceScalar(&lc, 
    											nd4j::indexreduce::IndexAbsoluteMax, 
    											nullptr, x3.getShapeInfo(),
    	                                       	dX3, dX3ShapeInfo, 
    	                                       	nullptr, 
    	                                       	nullptr, scalar.getShapeInfo(),
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

	 if (!Environment::getInstance()->isExperimentalBuild())
        return;

    NDArray x1('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
    NDArray x2('c', {2,2}, {-1,-2,-3,-4}, nd4j::DataType::INT32);
    NDArray x3('c', {2,2}, {1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
    NDArray x4('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);

    NDArray exp1('c', {}, {-30.f}, nd4j::DataType::FLOAT32);
    NDArray exp2('c', {}, {15.}, nd4j::DataType::DOUBLE);
    
	NDArray scalar1('c', {}, {100.f}, nd4j::DataType::FLOAT32);
    NDArray scalar2('c', {}, {100.}, nd4j::DataType::DOUBLE);

    void *dX1, *dX2, *dX3, *dX4, *dZ1, *dZ2; 
    Nd4jLong *dX1ShapeInfo, *dX3ShapeInfo, *dZ1ShapeInfo, *dZ2ShapeInfo;

    cudaError_t cudaResult;

    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1), x1.lengthOf() * x1.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2), x2.lengthOf() * x2.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3), x3.lengthOf() * x3.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX4), x4.lengthOf() * x4.sizeOfT()); 		   		         	 	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1), scalar1.lengthOf() * scalar1.sizeOfT());			         	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2), scalar2.lengthOf() * scalar2.sizeOfT());			         	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1ShapeInfo), shape::shapeInfoByteLength(x1.getShapeInfo()));    	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX3ShapeInfo), shape::shapeInfoByteLength(x3.getShapeInfo()));    	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1ShapeInfo), shape::shapeInfoByteLength(scalar1.getShapeInfo())); 	ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2ShapeInfo), shape::shapeInfoByteLength(scalar2.getShapeInfo())); 	ASSERT_EQ(0, cudaResult);

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
	cudaMemcpyAsync(dX1ShapeInfo, x1.getShapeInfo(), shape::shapeInfoByteLength(x1.getShapeInfo()), cudaMemcpyHostToDevice, stream);	
	cudaMemcpyAsync(dX3ShapeInfo, x3.getShapeInfo(), shape::shapeInfoByteLength(x3.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZ1ShapeInfo, scalar1.getShapeInfo(), shape::shapeInfoByteLength(scalar1.getShapeInfo()), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dZ2ShapeInfo, scalar2.getShapeInfo(), shape::shapeInfoByteLength(scalar2.getShapeInfo()), cudaMemcpyHostToDevice, stream);

	/***************************************/

	void* reductionPointer  = nullptr;
	int*  allocationPointer = nullptr;	

	cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024);		ASSERT_EQ(0, cudaResult);
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024);		ASSERT_EQ(0, cudaResult);

	LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);

	/***************************************/
	
    NativeOpExecutioner::execReduce3Scalar(&lc, nd4j::reduce3::Dot,nullptr, x1.getShapeInfo(),dX1, dX1ShapeInfo, nullptr, nullptr, x2.getShapeInfo(),dX2, dX1ShapeInfo,nullptr, scalar1.getShapeInfo(),dZ1, dZ1ShapeInfo);

    cudaResult = cudaStreamSynchronize(stream);     
    ASSERT_EQ(0, cudaResult);

    scalar1.tickWriteHost();
    scalar2.tickWriteHost();

    cudaMemcpyAsync(scalar1.buffer(), dZ1, scalar1.lengthOf() * scalar1.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream); 
    ASSERT_EQ(0, cudaResult);

	ASSERT_NEAR(exp1.e<float>(0), scalar1.e<float>(0), 1e-5);

    /***************************************/
    
    NativeOpExecutioner::execReduce3Scalar(&lc, nd4j::reduce3::Dot,nullptr, x3.getShapeInfo(),dX3, dX3ShapeInfo, nullptr, nullptr, x4.getShapeInfo(),dX4, dX3ShapeInfo,nullptr, scalar2.getShapeInfo(),dZ2, dZ2ShapeInfo);

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

    NDArray x('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
    NDArray y('c', {2,2}, {-1,-2,-3,-4}, nd4j::DataType::INT32);

    NDArray exp('c', {}, {-30.f}, nd4j::DataType::FLOAT32);
    NDArray z('c', {}, {100.f},  nd4j::DataType::FLOAT32);

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
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    
	NDArray x('c', {2,2}, {1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);

    NDArray exp('c', {}, {15.}, nd4j::DataType::DOUBLE);
    NDArray z('c', {}, {100.},  nd4j::DataType::DOUBLE);
   
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
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    
	NDArray x('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::INT32);
    NDArray y('c', {2,3}, {-6,-5,-4,-3,-2,-1}, nd4j::DataType::INT32);        

    NDArray exp('c', {3}, {-18,-20,-18}, nd4j::DataType::FLOAT32);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::FLOAT32);
   
    std::vector<int> dimensions = {0};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,3}, {1.5,1.5,1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);

    NDArray exp('c', {2}, {9,22.5}, nd4j::DataType::DOUBLE);
    NDArray z('c', {2}, {100,100}, nd4j::DataType::DOUBLE);
   
    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5}, nd4j::DataType::FLOAT32);
    NDArray y('c', {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, nd4j::DataType::FLOAT32);

    NDArray exp('c', {2,3}, {7.5, 10.5, 13.5, 25.5, 28.5, 31.5}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::FLOAT32);
   
    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
    NDArray y('c', {2,3}, {-1,1,-1,1,-1,1}, nd4j::DataType::INT32);

    NDArray exp('c', {2,3}, {2,-2,2,2,-2,2}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::FLOAT32);
   
    std::vector<int> dimensions = {0};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3All(&lc, nd4j::reduce3::Dot, 
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,3}, {1.5,1.5,1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);    

    NDArray exp('c', {2,3}, {6,6,6,9,9,9}, nd4j::DataType::DOUBLE);    
    NDArray z('c', {2,3}, {100,100,100,100,100,100,},nd4j::DataType::DOUBLE);    
   
    std::vector<int> dimensions = {0};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    // evaluate yTad data
    shape::TAD yTad;
    yTad.init(y.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3All(&lc, nd4j::reduce3::Dot, 
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
    x.linspace(-2.); x.syncToDevice();
    NDArray exp('c', {2}, {2, 2}, nd4j::DataType::INT64);
    NDArray z('c', {2}, {100,100}, nd4j::DataType::INT64);
    
    std::vector<int> dimensions = {1};          

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execIndexReduce(&lc, nd4j::indexreduce::IndexMax, 
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, 
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::FLOAT32);
    x.linspace(-2.f); x.syncToDevice();
    NDArray exp('c', {2,5}, {11,11,11,11,11,11,11,11,11,11}, nd4j::DataType::INT64);    
    NDArray z('c', {2,5}, {100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT64);
    
    std::vector<int> dimensions = {1,2};     

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execIndexReduce(&lc, nd4j::indexreduce::IndexMax, 
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, 
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    							100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
    x.linspace(-2.); x.syncToDevice();
    NDArray exp('c', {3}, {39, 39, 39}, nd4j::DataType::INT64);    
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::INT64);
    
    std::vector<int> dimensions = {0,2,3};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execIndexReduce(&lc, nd4j::indexreduce::IndexMax, 
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, 
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
    NDArray x('c', {2,3},  {0,1,2,3,4,5}, nd4j::DataType::INT64); 
    NDArray exp('c',{2,3}, {0,0,1,1,2,2}, nd4j::DataType::INT64);
    NDArray scalar('c',{}, {2.f}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::INT64);
    
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	
	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalar(&lc, nd4j::scalar::Divide, 
									nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
									nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
									nullptr, scalar.getShapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), 
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
    NDArray x('c', {2,3},  {-1,-2,-3,-4,-5,-6}, nd4j::DataType::INT64); 
    NDArray exp('c',{2,3}, {10,10,10,10,10,10}, nd4j::DataType::FLOAT32);
    NDArray scalar('c',{}, {10.f}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::FLOAT32);
    
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
		
	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalar(&lc, nd4j::scalar::CopyPws, 
									nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
									nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
									nullptr, scalar.getShapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), 
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
    NDArray x('c', {2,3,2},  {0,1,2,3,4,5,6,7,8,9,10,11}, nd4j::DataType::INT64); 
    NDArray scalars('c',{2,2}, {1,2,3,4}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,3,2},  {0,0,2,1,4,2, 2,1,2,2,3,2}, nd4j::DataType::INT64);     
    NDArray z('c', {2,3,2}, {100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT64);

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execScalar(&lc, nd4j::scalar::Divide, 
									nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
									nullptr,
									nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalars.getShapeInfo(), scalars.specialBuffer(), scalars.specialShapeInfo(),
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
    	
    NDArray x('c', {2,3},  {-1,-2,0,1,2,3}, nd4j::DataType::BFLOAT16); 
    NDArray scalar('c',{}, {0}, nd4j::DataType::BFLOAT16);
    NDArray exp('c',{2,3}, {0,0,0,1,1,1}, nd4j::DataType::BOOL);    
    NDArray z('c', {2,3}, {100,100,100,100,100,100,}, nd4j::DataType::BOOL);    
	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
		
	// call cuda kernel which calculates result
	// call cuda kernel which calculates result
	NativeOpExecutioner::execScalarBool(&lc, nd4j::scalar::GreaterThan, 
									nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
									nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
									nullptr, scalar.getShapeInfo(), scalar.specialBuffer(), scalar.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,3},  {0,1,2,3,4,5}, nd4j::DataType::FLOAT32); 
    NDArray scalars('c',{2}, {-1,4}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,3},  {1,1,1,0,0,1}, nd4j::DataType::BOOL);
    NDArray z('c', {2,3}, {100,100,100,100,100,100}, nd4j::DataType::BOOL);

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execScalarBool(&lc, nd4j::scalar::GreaterThan, 
									nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
									nullptr,
									nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
									nullptr, scalars.getShapeInfo(), scalars.specialBuffer(), scalars.specialShapeInfo(),
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);
    NDArray y('c', {3},   {10, 20, 30}, nd4j::DataType::INT64);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);	
	NDArray exp('c', {2,3,4}, {10, 11, 12, 13,24, 25, 26, 27,38, 39, 40, 41,22, 23, 24, 25,36, 37, 38, 39,50, 51, 52, 53}, nd4j::DataType::INT32);
	x.linspace(0); x.syncToDevice();

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execBroadcast(&lc, nd4j::broadcast::Add,
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);
    NDArray y('c', {2,4},   {10,20,30,40,50,60,70,80}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::FLOAT32);	
	NDArray exp('c', {2,3,4}, {10., 21., 32., 43., 14., 25., 36., 47., 18., 29., 40., 51., 62., 73., 84., 95., 66., 77., 88., 99., 70., 81., 92., 103}, nd4j::DataType::FLOAT32);
	x.linspace(0); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execBroadcast(&lc, nd4j::broadcast::Add,
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);
    NDArray y('c', {3},   {2, 12, 22}, nd4j::DataType::INT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,}, nd4j::DataType::BOOL);	
	NDArray exp('c', {2,3,4}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, nd4j::DataType::BOOL);
	x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execBroadcastBool(&lc, nd4j::broadcast::EqualTo,
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100},nd4j::DataType::FLOAT32);
    NDArray y('c', {2,4},   {1,10,10,15,20,20,20,24}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,3,4}, {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::BOOL);	
	NDArray exp('c', {2,3,4}, {1, 0, 0, 0,0, 0, 0, 0,0, 1, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 1}, nd4j::DataType::BOOL);
	x.linspace(1); x.syncToDevice();

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execBroadcastBool(&lc, nd4j::broadcast::EqualTo,
										nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
										nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
										nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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

	if (!Environment::getInstance()->isExperimentalBuild())
        return;
    	
	NDArray x('c', {2,2,2}, {1,5,3,7,2,6,4,8}, nd4j::DataType::INT32);
    NDArray y('c', {4,2}, {0.1,0.2,0.3,0.4,1.5,0.6,0.7,1.8}, nd4j::DataType::DOUBLE);
    NDArray z('c', {8}, {100,100,100,100,100,100,100,100}, nd4j::DataType::INT32);	
	NDArray exp('c', {8}, {0,1,2,3,3,5,6,6}, nd4j::DataType::INT32);
	x.permutei({2,1,0});	// -> {1,2,3,4,5,6,7,8}
    x.syncShape();

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	
	// call cuda kernel which calculates result
	NativeOpExecutioner::execPairwiseTransform(&lc, nd4j::pairwise::Subtract,
												nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
												nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
												nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
	NDArray x('c', {2,2,2}, {1,5,3,7,2,6,4,8}, nd4j::DataType::INT64);
    NDArray y('c', {4,2}, {0,2,0,4,0,6,0,8}, nd4j::DataType::INT64);
    NDArray z('c', {8}, {100,100,100,100,100,100,100,100}, nd4j::DataType::BOOL);	
	NDArray exp('c', {8}, {0,1,0,1,0,1,0,1}, nd4j::DataType::BOOL);
	x.permutei({2,1,0});	// -> {1,2,3,4,5,6,7,8}
	x.syncShape();
        
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);	

	// call cuda kernel which calculates result
	NativeOpExecutioner::execPairwiseBoolTransform(&lc, nd4j::pairwise::EqualTo,
													nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
													nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
													nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
	NDArray x('c', {2,2}, {0, 6.25, 2.25, 12.25}, nd4j::DataType::DOUBLE);    
    NDArray z('c', {4}, {100,100,100,100}, nd4j::DataType::FLOAT32);	
	NDArray exp('c', {4}, {0, 1.5, 2.5, 3.5}, nd4j::DataType::FLOAT32);
	x.permutei({1,0});
	x.syncShape();
        
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformFloat(&lc, nd4j::transform::Sqrt,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {1,4}, {0, 4, 9, 16}, nd4j::DataType::INT64);
    NDArray z('c', {2,2}, {100,100,100,100}, nd4j::DataType::DOUBLE);	
	NDArray exp('c', {2,2}, {0, 2, 3, 4}, nd4j::DataType::DOUBLE);	       
	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	
	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformFloat(&lc, nd4j::transform::Sqrt,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {2,2}, {0, 6.25, 2.25, 12.25}, nd4j::DataType::DOUBLE);    
    NDArray z('c', {4,1}, {100,100,100,100}, nd4j::DataType::INT32);	
	NDArray exp('c', {4,1}, {0, 2, 6, 12}, nd4j::DataType::INT32);
	x.permutei({1,0});
        
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformAny(&lc, nd4j::transform::Assign,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {1,4}, {0, 6.25, 2.25, 12.25}, nd4j::DataType::BFLOAT16);
    NDArray z('c', {2,2}, {100,100,100,100}, nd4j::DataType::FLOAT32);	
	NDArray exp('c', {2,2}, {0, 6.25, 2.25, 12.25}, nd4j::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformAny(&lc, nd4j::transform::Assign,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {2,3}, {0,2,4,1,3,5}, nd4j::DataType::DOUBLE);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, nd4j::DataType::DOUBLE);	
	NDArray exp('c', {3,2}, {0, 3, 12, 27, 48, 75}, nd4j::DataType::DOUBLE);
	x.permutei({1,0});
	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformStrict(&lc, nd4j::transform::CubeDerivative,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {6}, {0,1,2,3,4,5}, nd4j::DataType::FLOAT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, nd4j::DataType::FLOAT32);	
	NDArray exp('c', {3,2}, {0, 3, 12, 27, 48, 75}, nd4j::DataType::FLOAT32);	
    	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformStrict(&lc, nd4j::transform::CubeDerivative,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    
	NDArray x('c', {2,3}, {0,2.5,4.5,1.5,3.5,5.5}, nd4j::DataType::DOUBLE);	
    NDArray z('c', {1,6}, {100,100,100,100,100,100}, nd4j::DataType::DOUBLE);	
	NDArray exp('c', {1,6}, {0,2.25,6.25,12.25,20.25,30.25}, nd4j::DataType::DOUBLE);
	x.permutei({1,0});
    	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformSame(&lc, nd4j::transform::Square,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {6}, {0,1,2,3,4,5}, nd4j::DataType::INT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, nd4j::DataType::INT32);	
	NDArray exp('c', {3,2}, {0,1,4,9,16,25}, nd4j::DataType::INT32);	
    	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	
	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformSame(&lc, nd4j::transform::Square,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    
	NDArray x('c', {2,3}, {0,2,4,-1,-3,-5}, nd4j::DataType::DOUBLE);	
    NDArray z('c', {1,6}, {100,100,100,100,100,100}, nd4j::DataType::BOOL);	    
	NDArray exp('c', {1,6}, {0,0,1,0,1,0}, nd4j::DataType::BOOL);
	x.permutei({1,0});
    
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformBool(&lc, nd4j::transform::IsPositive,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	
	NDArray x('c', {6}, {0,-1,2,-3,4,-5}, nd4j::DataType::INT32);
    NDArray z('c', {3,2}, {100,100,100,100,100,100}, nd4j::DataType::BOOL);	
	NDArray exp('c', {3,2}, {0,0,1,0,1,0}, nd4j::DataType::BOOL);
    	
	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);

	// call cuda kernel which calculates result
	NativeOpExecutioner::execTransformBool(&lc, nd4j::transform::IsPositive,
		nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
		nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {3}, {2.5, 6.5, 10.5}, nd4j::DataType::FLOAT32);
    x.permutei({2,1,0});
    
    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceFloat(&lc, nd4j::reduce::Mean, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceFloat_2) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,4}, {-1., 0., 1., 2.,11., 12., 13., 14.}, nd4j::DataType::DOUBLE);
    
    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceFloat(&lc, nd4j::reduce::Mean, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceSame_1) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::INT32);
    NDArray exp('c', {3}, {20, 52, 84}, nd4j::DataType::INT32);
    x.permutei({2,1,0});
    
    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceSame(&lc, nd4j::reduce::Sum, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceSame_2) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,4}, {-3., 0., 3., 6.,33., 36., 39., 42.}, nd4j::DataType::FLOAT32);
    
    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceSame(&lc, nd4j::reduce::Sum, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceBool_1) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, nd4j::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::BOOL);
    NDArray exp('c', {3}, {0, 1, 1}, nd4j::DataType::BOOL);
    x.permutei({2,1,0});

    
    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceBool(&lc, nd4j::reduce::IsPositive, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceBool_2) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, nd4j::DataType::BOOL);
    NDArray exp('c', {2,4}, {1, 1, 1, 1, 0, 0, 0, 0}, nd4j::DataType::BOOL);
    
    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceBool(&lc, nd4j::reduce::IsPositive, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceLong_1) {
    	   	
    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, nd4j::DataType::INT32);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::INT64);
    NDArray exp('c', {3}, {5,6,6}, nd4j::DataType::INT64);
    x.permutei({2,1,0});
    
    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceLong(&lc, nd4j::reduce::CountNonZero, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceLong_2) {
    	   	
    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,4}, {100,100,100,100,100,100,100,100}, nd4j::DataType::INT64);
    NDArray exp('c', {2,4}, {3, 1, 3, 2, 2, 1, 2, 3}, nd4j::DataType::INT64);    

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduceLong(&lc, nd4j::reduce::CountNonZero, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
TEST_F(CudaBasicsTests1, execReduceFloatScalar_1) {
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {}, {100}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {}, {6.5}, nd4j::DataType::FLOAT32);
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
	NativeOpExecutioner::execReduceFloatScalar(&lc, nd4j::reduce::Mean, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());
	
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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {}, {100}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {}, {6.5}, nd4j::DataType::DOUBLE);
	
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
	NativeOpExecutioner::execReduceFloatScalar(&lc, nd4j::reduce::Mean, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());

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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::INT32);
    NDArray z('c', {}, {100}, nd4j::DataType::INT32);
    NDArray exp('c', {}, {156}, nd4j::DataType::INT32);
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
	NativeOpExecutioner::execReduceSameScalar(&lc, nd4j::reduce::Sum, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());
	
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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, nd4j::DataType::DOUBLE);
    NDArray z('c', {}, {100}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {}, {156}, nd4j::DataType::DOUBLE);
	
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
	NativeOpExecutioner::execReduceSameScalar(&lc, nd4j::reduce::Sum, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());

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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, nd4j::DataType::INT32);
    NDArray z('c', {}, {100}, nd4j::DataType::BOOL);
    NDArray exp('c', {}, {1}, nd4j::DataType::BOOL);
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
	NativeOpExecutioner::execReduceBoolScalar(&lc, nd4j::reduce::IsPositive, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());
	
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
    	   	
    NDArray x('c', {2,3,4}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18}, nd4j::DataType::DOUBLE);
    NDArray z('c', {}, {100}, nd4j::DataType::BOOL);
    NDArray exp('c', {}, {1}, nd4j::DataType::BOOL);
    
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
	NativeOpExecutioner::execReduceBoolScalar(&lc, nd4j::reduce::IsPositive, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());

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
    	   	
    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, nd4j::DataType::INT32);
    NDArray z('c', {}, {100}, nd4j::DataType::INT64);
    NDArray exp('c', {}, {17}, nd4j::DataType::INT64);
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
	NativeOpExecutioner::execReduceLongScalar(&lc, nd4j::reduce::CountNonZero, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());
	
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
    	   	
    NDArray x('c', {2,3,4}, {-5,0,-3,0,-1,0,1,2,3,4,5,6,7,0,9,10,11,0,13,14,0,16,0,18}, nd4j::DataType::DOUBLE);
    NDArray z('c', {}, {100}, nd4j::DataType::INT64);
    NDArray exp('c', {}, {17}, nd4j::DataType::INT64);
    
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
	NativeOpExecutioner::execReduceLongScalar(&lc, nd4j::reduce::CountNonZero, 
					nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
					nullptr, 
					nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo());

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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::FLOAT32);
    NDArray y('c', {2,2}, {1,2,3,4}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {3}, {10,20,30}, nd4j::DataType::DOUBLE);
    NDArray z('c', {3}, {100,100,100}, nd4j::DataType::DOUBLE);
   
    std::vector<int> dimensions = {0,1};
    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), dimensions);
    LaunchContext* context = x.getContext();

	x.syncToDevice();
	y.syncToDevice();
	PointersManager pm(context, "execReduce3TAD_1");
	// call cuda kernel which calculates result
	NativeOpExecutioner::execReduce3TAD(context, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::INT64);
    NDArray y('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::INT64);
    NDArray exp('c', {2}, {10,73}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, nd4j::DataType::FLOAT32);
   
    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3TAD(&lc, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::INT64);
    NDArray y('c', {3}, {1,2,3}, nd4j::DataType::INT64);
    NDArray exp('c', {2,2}, {-22,-4,14,32}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,2}, {100,100,100,100}, nd4j::DataType::FLOAT32);
   
    std::vector<int> dimensions = {2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3TAD(&lc, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,2,3}, {10,20,30,40,50,60,70,80,90,100,110,120}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {}, {1820}, nd4j::DataType::FLOAT32);
    NDArray z('c', {}, {100}, nd4j::DataType::FLOAT32);

    std::vector<int> dimensions = {0,1,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execReduce3TAD(&lc, nd4j::reduce3::Dot,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 
								nullptr, y.getShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::INT64);    
    NDArray exp('c', {}, {3.605551}, nd4j::DataType::FLOAT32);
    NDArray z('c', {}, {100}, nd4j::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024); ASSERT_EQ(0, cudaResult);    	
	lc.setReductionPointer(reductionPointer);
		
	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStats(&lc, nd4j::variance::SummaryStatsStandardDeviation,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 								
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-20,-1,0,1,2,3,4,5,6}, nd4j::DataType::DOUBLE);    
    NDArray exp('c', {2}, {3.405877, 9.715966}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, nd4j::DataType::FLOAT32);

    std::vector<int> dimensions = {0,2};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execSummaryStats(&lc, nd4j::variance::SummaryStatsStandardDeviation,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 								
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStats_3) {
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-20,-1,0,1,2,3,4,5,6}, nd4j::DataType::DOUBLE);    
    NDArray exp('c', {2}, {10.606602, 2.121320}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2}, {100,100}, nd4j::DataType::FLOAT32);

    std::vector<int> dimensions = {1};

    // evaluate xTad data 
    shape::TAD xTad;
    xTad.init(x.getShapeInfo(), dimensions.data(), dimensions.size());
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
	NativeOpExecutioner::execSummaryStats(&lc, nd4j::variance::SummaryStatsStandardDeviation,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 								
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests1, execSummaryStatsScalar_1) {
    	
    NDArray x('c', {2,2,3}, {-5,-4,-3,-2,-1,0,1,2,3,4,5,6}, nd4j::DataType::INT64);
    NDArray exp('c', {}, {3.605551}, nd4j::DataType::FLOAT32);
    NDArray z('c', {}, {100}, nd4j::DataType::FLOAT32);

	// create cuda stream and LaunchContext
	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
	LaunchContext lc(&stream);
	void* reductionPointer;
    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024); ASSERT_EQ(0, cudaResult);    	
	lc.setReductionPointer(reductionPointer);
		
	// call cuda kernel which calculates result
	NativeOpExecutioner::execSummaryStatsScalar(&lc, nd4j::variance::SummaryStatsStandardDeviation,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, 								
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
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
    	   
//    NDArray z('c', {10}, {100,0,0,0,0,0,0,0,0,0}, nd4j::DataType::DOUBLE);
    NDArray z('c', {10}, {100,0,0,0,0,0,0,0,0,100}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {10}, {0.050942, -0.183229, -0.093921, 0.075469, 0.257166, -0.254838, 0.342227, -0.682188, -0.004345, 0.464633}, nd4j::DataType::FLOAT32);

    nd4j::graph::RandomGenerator gen(119,5);

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
//	//	::execRandom(extraPointers, random::GaussianDistribution, &gen, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), &extra);
//	// call cuda kernel which calculates result
//	NativeOpExecutioner::execRandom(&lc, nd4j::random::GaussianDistribution,
//								&gen,
//								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
//								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
//								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
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
    	   
    NDArray x('c', {10}, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}, nd4j::DataType::DOUBLE);    
    NDArray z('c', {2,5}, {100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {10}, {0., 0., 0.3, 0., 0.5, 0., 0.7, 0., 0., 1.}, nd4j::DataType::DOUBLE);
    
    ExtraArguments extraArguments({0.7});
    nd4j::graph::RandomGenerator gen(119,5);
    
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
	NativeOpExecutioner::execRandom(lc, nd4j::random::DropOut,
								&gen,
								nullptr, x.getShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 								
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
    	       
    NDArray z('c', {10}, {100,100,100,100,100,100,100,100,100,100}, nd4j::DataType::DOUBLE);    
    NDArray exp('c', {10}, {2.373649, 2.239791, 1.887353, 2.488636, 2.068904, 2.281399, 1.828228, 2.228222, 2.490847, 1.669537}, nd4j::DataType::DOUBLE);
    
    std::vector<double> extraArguments = {1.5, 2.5};
    nd4j::graph::RandomGenerator gen(119,5);
    
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
	NativeOpExecutioner::execRandom(&lc, nd4j::random::UniformDistribution,
								&gen,
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 								
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
    	       
    NDArray z('c', {2,5}, {1,2,3,4,5,6,7,8,9,10}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {10}, {2.373649, 2.281399, 2.239791, 1.828228, 1.887353, 2.228222, 2.488636, 2.490847, 2.068904, 1.669537}, nd4j::DataType::FLOAT32);
    z.permutei({1,0});        
        
    ExtraArguments extraArguments({1.5, 2.5});
    nd4j::graph::RandomGenerator gen(119,5);
    
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
	NativeOpExecutioner::execRandom(context, nd4j::random::UniformDistribution,
								&gen,
								nullptr, z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 								
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

