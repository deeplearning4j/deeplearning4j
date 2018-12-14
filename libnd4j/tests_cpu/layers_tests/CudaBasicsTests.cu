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
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>

#include <cuda.h>
#include <cuda_launch_config.h>

using namespace nd4j;
using namespace nd4j::graph;

class CudaBasicsTests : public testing::Test {
public:

};



TEST_F(CudaBasicsTests, TestPairwise_1) {
	// allocating host-side arrays
	auto x = NDArrayFactory::create<double>('c', { 5 }, { 1, 2, 3, 4, 5});
	auto z = NDArrayFactory::create<double>('c', { 5 });

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
	CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream");
	cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
	auto stream = reinterpret_cast<cudaStream_t *>(&nativeStream);

	cudaMemcpyAsync(devBufferPtrX, x.buffer(), x.lengthOf() * x.sizeOfT(), cudaMemcpyHostToDevice, *stream);
	cudaMemcpyAsync(devShapePtrX, x.shapeInfo(), shape::shapeInfoByteLength(x.shapeInfo()), cudaMemcpyHostToDevice, *stream);
	
	LaunchContext lc(stream, nullptr, nullptr);
	NativeOpExecutioner::execPairwiseTransform(&lc, pairwise::Add, nullptr, x.shapeInfo(), devBufferPtrX, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr, x.shapeInfo(), devBufferPtrX, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr, z.shapeInfo(), devBufferPtrZ, reinterpret_cast<Nd4jLong*>(devShapePtrX), nullptr);
	res = cudaStreamSynchronize(*stream);
	ASSERT_EQ(0, res);


	cudaMemcpyAsync(z.buffer(), devBufferPtrZ, z.lengthOf() * x.sizeOfT(), cudaMemcpyDeviceToHost, *stream);
	res = cudaStreamSynchronize(*stream);
	ASSERT_EQ(0, res);

	cudaFree(devBufferPtrX);
	cudaFree(devBufferPtrZ);
	cudaFree(devShapePtrX);

	for (int e = 0; e < z.lengthOf(); e++) {
		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);
	}
}


////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests, execIndexReducescalar_1) {

    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray x2('c', {2,2}, {0.5, 1.5, -4.5, 3.5}, nd4j::DataType::HALF);    
    NDArray x3('c', {2,2}, {0, -1, 0, 1}, nd4j::DataType::BOOL);
    
    NDArray scalar(nd4j::DataType::INT64);

    NDArray exp1('c', {0}, {3}, nd4j::DataType::INT64);
    NDArray exp2('c', {0}, {2}, nd4j::DataType::INT64);
    NDArray exp3('c', {0}, {1}, nd4j::DataType::INT64);

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

	LaunchContext lc(&stream, reductionPointer);

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