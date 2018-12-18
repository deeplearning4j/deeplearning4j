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


//////////////////////////////////////////////////////////////////////////
static cudaError_t prepareDataForCuda(cudaStream_t& stream, void* reductionPointer, int* allocationPointer,
										std::vector<NDArray*>& arrs,
										std::vector<void*>& dBuffs,
										std::vector<Nd4jLong*>& dShapes,
										std::vector<int>& dimensions,
										int *dDimensions,
										std::vector<Nd4jLong*>& tadOnlyShapeInfo,
										std::vector<Nd4jLong*>& tadOffsets) { 
    

	cudaError_t cudaResult;

    cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer),  1024*1024);		if(cudaResult != 0) return cudaResult;
	cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024);		if(cudaResult != 0) return cudaResult;

	// allocating device memory for arrays	
	for(int i = 0; i < arrs.size(); ++i) {					

		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dBuffs[i]), arrs[i]->lengthOf() * arrs[i]->sizeOfT()); 				if(cudaResult != 0) return cudaResult;
		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dShapes[i]), shape::shapeInfoByteLength(arrs[i]->getShapeInfo()));   if(cudaResult != 0) return cudaResult;

		cudaMemcpyAsync(dShapes[i], arrs[i]->getShapeInfo(), shape::shapeInfoByteLength(arrs[i]->getShapeInfo()), cudaMemcpyHostToDevice, stream);
		if(i != arrs.size()-1)	// do not copy buffer for result last array
			cudaMemcpyAsync(dBuffs[i], arrs[i]->buffer(), arrs[i]->lengthOf() * arrs[i]->sizeOfT(),  cudaMemcpyHostToDevice, stream);			
	}
	
    // evaluating and allocating device memory for tad
    int dimensionsLength = dimensions.size();    
    for(int i = 0; i < tadOnlyShapeInfo.size(); ++i) {

    	shape::TAD tad(arrs[i]->getShapeInfo(), dimensions.data(), dimensionsLength);    	    
    	tad.createTadOnlyShapeInfo();
    	tad.createOffsets();

    	Nd4jLong* tadShapeInfo = tad.tadOnlyShapeInfo;
		Nd4jLong* tadSteps 	   = tad.tadOffsets;

		cudaResult = cudaMalloc(reinterpret_cast<void **>(&tadOnlyShapeInfo[i]), shape::shapeInfoByteLength(tadShapeInfo));	if(cudaResult != 0) return cudaResult;
		cudaResult = cudaMalloc(reinterpret_cast<void **>(&tadOffsets[i]), tad.numTads * sizeof(Nd4jLong));					if(cudaResult != 0) return cudaResult;

		cudaMemcpyAsync(tadOnlyShapeInfo[i], tadShapeInfo, shape::shapeInfoByteLength(tadShapeInfo), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(tadOffsets[i], tadSteps, tad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, stream);
    }

    // allocate device memory for dimensions
    if(dimensionsLength != 0) {
		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dDimensions), dimensionsLength * sizeof(int));					if(cudaResult != 0) return cudaResult;
    	cudaMemcpyAsync(dDimensions, dimensions.data(), dimensionsLength * sizeof(int), cudaMemcpyHostToDevice, stream);
    }    

    int temp1[2];
    cudaMemcpyAsync(temp1, dBuffs[0], arrs[0]->lengthOf() * arrs[0]->sizeOfT(), cudaMemcpyDeviceToHost, stream);    
    for (int i = 0; i < 2; ++i)
    	printf("%i, ", temp1[i]);
    printf("\n");

    cudaMemcpyAsync(temp1, dBuffs[1], arrs[1]->lengthOf() * arrs[1]->sizeOfT(), cudaMemcpyDeviceToHost, stream);    
    for (int i = 0; i < 2; ++i)
    	printf("%i, ", temp1[i]);
    printf("\n");

	Nd4jLong temp2[8];
    cudaMemcpyAsync(temp2, dShapes[0], shape::shapeInfoByteLength(arrs[0]->getShapeInfo()), cudaMemcpyDeviceToHost, stream);    
    for (int i = 0; i < 8; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");

    cudaMemcpyAsync(temp2, dShapes[1], shape::shapeInfoByteLength(arrs[1]->getShapeInfo()), cudaMemcpyDeviceToHost, stream);
    for (int i = 0; i < 8; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");

    
    cudaMemcpyAsync(temp1, dDimensions, dimensionsLength * sizeof(int), cudaMemcpyDeviceToHost, stream);
  	for (int i = 0; i < 2; ++i)
    	printf("%i, ", temp1[i]);
    printf("\n");  

    cudaMemcpyAsync(temp2, tadOnlyShapeInfo[0], 8 * sizeof(Nd4jLong), cudaMemcpyDeviceToHost, stream);
  	for (int i = 0; i < 8; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");  

    cudaMemcpyAsync(temp2, tadOnlyShapeInfo[1], 8 * sizeof(Nd4jLong), cudaMemcpyDeviceToHost, stream);
  	for (int i = 0; i < 8; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");  
  
  	cudaMemcpyAsync(temp2, tadOffsets[0], sizeof(Nd4jLong), cudaMemcpyDeviceToHost, stream);
  	for (int i = 0; i < 1; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");  

    cudaMemcpyAsync(temp2, tadOffsets[1], sizeof(Nd4jLong), cudaMemcpyDeviceToHost, stream);
  	for (int i = 0; i < 1; ++i)
    	printf("%i, ", temp2[i]);
    printf("\n");  
    
	return cudaResult;
}


//////////////////////////////////////////////////////////////////////////
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
TEST_F(CudaBasicsTests, execIndexReduceScalar_1) {

    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray x2('c', {2,2}, {0.5, 1.5, -4.5, 3.5}, nd4j::DataType::BFLOAT16);    
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

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests, execReduce3Scalar_1) {

    NDArray x1('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
    NDArray x2('c', {2,2}, {-1,-2,-3,-4}, nd4j::DataType::INT32);
    NDArray x3('c', {2,2}, {1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
    NDArray x4('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {0}, {-30}, nd4j::DataType::FLOAT32);
    NDArray exp2('c', {0}, {15}, nd4j::DataType::DOUBLE);
    
	NDArray scalar1('c', {0}, nd4j::DataType::FLOAT32);
    NDArray scalar2('c', {0}, nd4j::DataType::DOUBLE);

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
TEST_F(CudaBasicsTests, execReduce3_1) {

    NDArray x('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
    NDArray y('c', {2,2}, {-1,-2,-3,-4}, nd4j::DataType::INT32);

    NDArray exp('c', {0}, {-30}, nd4j::DataType::FLOAT32);
    NDArray z('c', {0},   nd4j::DataType::FLOAT32);
        
    void *reductionPointer;
	int *dDimensions, *allocationPointer;	

	std::vector<NDArray*> arrs = {&x,&y,&z};
	std::vector<void*> dBuffs(3, nullptr);
	std::vector<Nd4jLong*> dShapes(3, nullptr);
	std::vector<int> dimensions = {0,1};
	std::vector<Nd4jLong*> tadOnlyShapeInfo(2, nullptr);
	std::vector<Nd4jLong*> tadOffsets(2, nullptr);

	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);
	ASSERT_EQ(0, cudaResult);

	cudaResult = prepareDataForCuda(stream, reductionPointer, allocationPointer, arrs, dBuffs, dShapes, dimensions, dDimensions, tadOnlyShapeInfo, tadOffsets);
	ASSERT_EQ(0, cudaResult);		

	LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);   

	/***************************************/
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
									nullptr, x.getShapeInfo(), dBuffs[0], dShapes[0], 
									nullptr, 
									nullptr, y.getShapeInfo(), dBuffs[1], dShapes[1], 
									nullptr, z.getShapeInfo(), dBuffs[2], dShapes[2], 
									dDimensions, 2, 
									tadOnlyShapeInfo[0], tadOffsets[0], tadOnlyShapeInfo[1], tadOffsets[1]);

	cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
    cudaMemcpyAsync(z.buffer(), dBuffs[0], z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
 	
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++) 
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	/***************************************/
	for(int i = 0; i < dBuffs.size(); ++i) 			 { cudaFree(dBuffs[i]); cudaFree(dShapes[i]);	}
	for(int i = 0; i < tadOnlyShapeInfo.size(); ++i) { cudaFree(tadOnlyShapeInfo[i]); cudaFree(tadOffsets[i]); }
	cudaFree(dDimensions);

	/***************************************/	

	cudaResult = cudaStreamDestroy(stream); 
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests, execReduce3_2) {

	NDArray x('c', {2,2}, {1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
    NDArray y('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);

    NDArray exp('c', {0}, {15}, nd4j::DataType::DOUBLE);
    NDArray z('c', {0},   nd4j::DataType::DOUBLE);
        
    void *reductionPointer;
	int *dDimensions, *allocationPointer;	

	std::vector<NDArray*> arrs = {&x,&y,&z};
	std::vector<void*> dBuffs(3, nullptr);
	std::vector<Nd4jLong*> dShapes(3, nullptr);
	std::vector<int> dimensions = {0,1};
	std::vector<Nd4jLong*> tadOnlyShapeInfo(2, nullptr);
	std::vector<Nd4jLong*> tadOffsets(2, nullptr);

	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);
	ASSERT_EQ(0, cudaResult);

	cudaResult = prepareDataForCuda(stream, reductionPointer, allocationPointer, arrs, dBuffs, dShapes, dimensions, dDimensions, tadOnlyShapeInfo, tadOffsets);
	ASSERT_EQ(0, cudaResult);		

	LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);   

	/***************************************/
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
									nullptr, x.getShapeInfo(), dBuffs[0], dShapes[0], 
									nullptr, 
									nullptr, y.getShapeInfo(), dBuffs[1], dShapes[1], 
									nullptr, z.getShapeInfo(), dBuffs[2], dShapes[2], 
									dDimensions, 2, 
									tadOnlyShapeInfo[0], tadOffsets[0], tadOnlyShapeInfo[1], tadOffsets[1]);

	cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
    cudaMemcpyAsync(z.buffer(), dBuffs[0], z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
 	
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++) 
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	/***************************************/
	for(int i = 0; i < dBuffs.size(); ++i) 			 { cudaFree(dBuffs[i]); cudaFree(dShapes[i]);	}
	for(int i = 0; i < tadOnlyShapeInfo.size(); ++i) { cudaFree(tadOnlyShapeInfo[i]); cudaFree(tadOffsets[i]); }
	cudaFree(dDimensions);

	/***************************************/	

	cudaResult = cudaStreamDestroy(stream); 
	ASSERT_EQ(0, cudaResult);
}

////////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests, execReduce3_3) {

	NDArray x('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::INT32);
    NDArray y('c', {2,3}, {-6,-5,-4,-3,-2,-1}, nd4j::DataType::INT32);        

    NDArray exp('c', {3}, {-18,-20,-18}, nd4j::DataType::FLOAT32);
    NDArray z('c', {3}, nd4j::DataType::FLOAT32);
        
    void *reductionPointer;
	int *dDimensions, *allocationPointer;	

	std::vector<NDArray*> arrs = {&x,&y,&z};
	std::vector<void*> dBuffs(3, nullptr);
	std::vector<Nd4jLong*> dShapes(3, nullptr);
	std::vector<int> dimensions = {0};
	std::vector<Nd4jLong*> tadOnlyShapeInfo(2, nullptr);
	std::vector<Nd4jLong*> tadOffsets(2, nullptr);

	cudaError_t cudaResult;
	cudaStream_t stream;
	cudaResult = cudaStreamCreate(&stream);
	ASSERT_EQ(0, cudaResult);

	cudaResult = prepareDataForCuda(stream, reductionPointer, allocationPointer, arrs, dBuffs, dShapes, dimensions, dDimensions, tadOnlyShapeInfo, tadOffsets);
	ASSERT_EQ(0, cudaResult);			

	LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);   

	/***************************************/
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
									nullptr, x.getShapeInfo(), dBuffs[0], dShapes[0], 
									nullptr, 
									nullptr, y.getShapeInfo(), dBuffs[1], dShapes[1], 
									nullptr, z.getShapeInfo(), dBuffs[2], dShapes[2], 
									dDimensions, 2, 
									tadOnlyShapeInfo[0], tadOffsets[0], tadOnlyShapeInfo[1], tadOffsets[1]);

	cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
    cudaMemcpyAsync(z.buffer(), dBuffs[0], z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
 	
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++) 
 		ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	/***************************************/

 	/***************************************/
	NativeOpExecutioner::execReduce3(&lc, nd4j::reduce3::Dot, 
									nullptr, x.getShapeInfo(), dBuffs[0], dShapes[0], 
									nullptr, 
									nullptr, y.getShapeInfo(), dBuffs[1], dShapes[1], 
									nullptr, z.getShapeInfo(), dBuffs[2], dShapes[2], 
									dDimensions, 2, 
									tadOnlyShapeInfo[0], tadOffsets[0], tadOnlyShapeInfo[1], tadOffsets[1]);

	cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
    cudaMemcpyAsync(z.buffer(), dBuffs[0], z.lengthOf() * z.sizeOfT(), cudaMemcpyDeviceToHost, stream);

    cudaResult = cudaStreamSynchronize(stream);
    ASSERT_EQ(0, cudaResult);
 	
 	// verify results
 	for (int e = 0; e < z.lengthOf(); e++) ASSERT_NEAR(exp.e<double>(e), z.e<double>(e), 1e-5);

	/***************************************/
	for(int i = 0; i < dBuffs.size(); ++i) 			 { cudaFree(dBuffs[i]); cudaFree(dShapes[i]);	}
	for(int i = 0; i < tadOnlyShapeInfo.size(); ++i) { cudaFree(tadOnlyShapeInfo[i]); cudaFree(tadOffsets[i]); }
	cudaFree(dDimensions);

	/***************************************/	

	cudaResult = cudaStreamDestroy(stream); 
	ASSERT_EQ(0, cudaResult);
}

// ////////////////////////////////////////////////////////////////////////////
// TEST_F(CudaBasicsTests, execReduce3_1) {

//     NDArray x1('c', {2,2}, {1,2,3,4}, nd4j::DataType::INT32);
//     NDArray x2('c', {2,2}, {-1,-2,-3,-4}, nd4j::DataType::INT32);    
//     NDArray x3('c', {2,2}, {1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
//     NDArray x4('c', {2,2}, {1,2,3,4}, nd4j::DataType::DOUBLE);
    
//     NDArray x5('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::INT32);
//     NDArray x6('c', {2,3}, {-6,-5,-4,-3,-2,-1}, nd4j::DataType::INT32);    
    
//     NDArray x7('c', {2,3}, {1.5,1.5,1.5,1.5,1.5,1.5}, nd4j::DataType::DOUBLE);
//     NDArray x8('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::DOUBLE);
//     NDArray x9('c', {2,2,3}, {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5}, nd4j::DataType::FLOAT32);
//     NDArray x10('c', {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, nd4j::DataType::FLOAT32);

//     NDArray exp1('c', {0}, {-30}, nd4j::DataType::FLOAT32);
//     NDArray exp2('c', {0}, {15}, nd4j::DataType::DOUBLE);
//     NDArray exp3('c', {3}, {-18,-20,-18}, nd4j::DataType::FLOAT32);
//     NDArray exp4('c', {2}, {-28,-28}, nd4j::DataType::FLOAT32);
//     NDArray exp5('c', {3}, {7.5,10.5,13.5}, nd4j::DataType::DOUBLE);
//     NDArray exp6('c', {2}, {9,22.5}, nd4j::DataType::DOUBLE);        
//     NDArray exp7('c', {2,3}, {7.5, 10.5, 13.5, 25.5, 28.5, 31.5}, nd4j::DataType::FLOAT32);

//     NDArray res1('c', {0},   nd4j::DataType::FLOAT32);
//     NDArray res2('c', {0},   nd4j::DataType::FLOAT32);
//     NDArray res3('c', {0},   nd4j::DataType::DOUBLE);
//     NDArray res4('c', {0},   nd4j::DataType::DOUBLE);
//     NDArray res5('c', {3},   nd4j::DataType::FLOAT32);
//     NDArray res6('c', {2},   nd4j::DataType::FLOAT32);
//     NDArray res7('c', {3},   nd4j::DataType::DOUBLE);
//     NDArray res8('c', {2},   nd4j::DataType::DOUBLE);    
//     NDArray res9('c', {2,3}, nd4j::DataType::FLOAT32);
//     NDArray res10('c', {2,3}, nd4j::DataType::FLOAT32);
   
//    	const int N = 10;
// 	NDArray x[N] = {x1, x2, x3, x4, x5, x6, x7, x8, x9, x10};
// 	NDArray exp[] = {exp1, exp1, exp2, exp2, exp3, exp4, exp5, exp6, exp7, exp7};
// 	NDArray res[] = {res1, res2, res3, res4, res5, res6, res7, res8, res9, res10};
// 	std::vector<std::vector<int>> dimensions = {{0,1},{0,1},  {0,1},{0,1},  {0},{1},   {0},{1},   {1},{1}};

//     for(int i = 2; i < 3; ++i) {
		
// 		void *dX1, *dX2, *dZ1, *dZ2;    	
// 		int *dDimensions, int 
// 		Nd4jLong *dXShapeInfo, *dZ1ShapeInfo, *dZ2ShapeInfo;

// 		cudaError_t cudaResult;

//     	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX1), x[2*i].lengthOf()     * x[2*i].sizeOfT()); 		   		    	ASSERT_EQ(0, cudaResult);
//     	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dX2), x[2*i+1].lengthOf()   * x[2*i+1].sizeOfT()); 		   				ASSERT_EQ(0, cudaResult);
//     	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1), res[2*i].lengthOf()   * res[2*i].sizeOfT());			        	ASSERT_EQ(0, cudaResult);
//     	cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2), res[2*i+1].lengthOf() * res[2*i+1].sizeOfT());			     		ASSERT_EQ(0, cudaResult);
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dXShapeInfo), shape::shapeInfoByteLength(x[2*i].getShapeInfo()));   		ASSERT_EQ(0, cudaResult);
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ1ShapeInfo), shape::shapeInfoByteLength(res[2*i].getShapeInfo())); 	ASSERT_EQ(0, cudaResult);
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dZ2ShapeInfo), shape::shapeInfoByteLength(res[2*i+1].getShapeInfo()));	ASSERT_EQ(0, cudaResult);
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&dDimensions), shape::shapeInfoByteLength(res[2*i+1].getShapeInfo()));	ASSERT_EQ(0, cudaResult);

// 		cudaStream_t stream;
// 		cudaResult = cudaStreamCreate(&stream); 		
// 		ASSERT_EQ(0, cudaResult);

// 		cudaMemcpyAsync(dX1, x[2*i].buffer(),   x[2*i].lengthOf()   * x[2*i].sizeOfT(),   cudaMemcpyHostToDevice, stream);
// 		cudaMemcpyAsync(dX2, x[2*i+1].buffer(), x[2*i+1].lengthOf() * x[2*i+1].sizeOfT(), cudaMemcpyHostToDevice, stream);
// 		cudaMemcpyAsync(dXShapeInfo, x[2*i].getShapeInfo(), shape::shapeInfoByteLength(x[2*i].getShapeInfo()), cudaMemcpyHostToDevice, stream);	
// 		cudaMemcpyAsync(dZ1ShapeInfo, res[2*i].getShapeInfo(), shape::shapeInfoByteLength(res[2*i].getShapeInfo()), cudaMemcpyHostToDevice, stream);
// 		cudaMemcpyAsync(dZ2ShapeInfo, res[2*i+1].getShapeInfo(), shape::shapeInfoByteLength(res[2*i+1].getShapeInfo()), cudaMemcpyHostToDevice, stream);
		
// 		void*  reductionPointer = nullptr;
// 		int*   allocationPointer = nullptr;		
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&reductionPointer), 1024*1024);		ASSERT_EQ(0, cudaResult);
// 		cudaResult = cudaMalloc(reinterpret_cast<void **>(&allocationPointer), 1024*1024);		ASSERT_EQ(0, cudaResult);
		

// 		LaunchContext lc(&stream, reductionPointer, nullptr, allocationPointer);

// 		NativeOpExecutioner::execReduce3(&lc, 
//     								nd4j::reduce3::Dot,
//     								nullptr, x[2*i].getShapeInfo(),
//     	                            dX1, dXShapeInfo, 
//     	                            nullptr, 
//     	                            nullptr, x[2*i+1].getShapeInfo(),
//     	                            dX2, dXShapeInfo,
//     	                            nullptr, res[2*i].getShapeInfo(),
//     	                            dZ1, dZ1ShapeInfo,
//     	                            dimensions[2*i].data(), dimensions[2*i].size());

// 		cudaResult = cudaStreamSynchronize(stream);
//     	ASSERT_EQ(0, cudaResult);
//     	cudaMemcpyAsync(res[2*i].buffer(), dZ1, res[2*i].lengthOf() * res[2*i].sizeOfT(), cudaMemcpyDeviceToHost, stream);

//     	cudaResult = cudaStreamSynchronize(stream);
//     	ASSERT_EQ(0, cudaResult);
 		
//  		for (int e = 0; e < res[2*i].lengthOf(); e++)
// 			ASSERT_NEAR(exp[2*i].e<double>(e), res[2*i].e<double>(e), 1e-5);

// 		/***************************************/

// 		NativeOpExecutioner::execReduce3(&lc, 
//     								nd4j::reduce3::Dot,
//     								nullptr, x[2*i+1].getShapeInfo(),
//     	                            dX2, dXShapeInfo, 
//     	                            nullptr, 
//     	                            nullptr, x[2*i].getShapeInfo(),
//     	                            dX1, dXShapeInfo,    	                            
//     	                            nullptr, res[2*i+1].getShapeInfo(),
//     	                            dZ2, dZ2ShapeInfo,
//     	                            dimensions[2*i+1].data(), dimensions[2*i+1].size());

// 		cudaResult = cudaStreamSynchronize(stream);     
//     	ASSERT_EQ(0, cudaResult);
//     	cudaMemcpyAsync(res[2*i+1].buffer(), dZ2, res[2*i+1].lengthOf() * res[2*i+1].sizeOfT(), cudaMemcpyDeviceToHost, stream);

//     	cudaResult = cudaStreamSynchronize(stream); 
//     	ASSERT_EQ(0, cudaResult);
 		
//  		for (int e = 0; e < res[2*i+1].lengthOf(); e++)
// 			ASSERT_NEAR(exp[2*i+1].e<double>(e), res[2*i+1].e<double>(e), 1e-5);

// 		/***************************************/
// 		cudaFree(dX1); 			cudaFree(dX2);  cudaFree(dZ1); 			cudaFree(dZ2);
// 		cudaFree(dXShapeInfo);					cudaFree(dZ1ShapeInfo);	cudaFree(dZ2ShapeInfo);

// 		/***************************************/	

// 		cudaResult = cudaStreamDestroy(stream); 
// 		ASSERT_EQ(0, cudaResult);
//     }
// }
