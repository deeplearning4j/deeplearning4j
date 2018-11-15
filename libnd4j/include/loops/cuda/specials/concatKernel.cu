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
// @author Yurii Shyrma, created on 15.11.2018
//

#include <loops/special_kernels.h>


///////////////////////////////////////////////////////////////////////
template <typename T>
__device__ void concatKernel(int dimension,
							int numArrays,
							Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
							void *vz, Nd4jLong *zShapeInfo, 
							Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, 
							Nd4jLong *zTadShape, 
							Nd4jLong *zOffsets) {
	
	auto z = static_cast<T*>(vz);
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int zRank = shape::rank(zShapeInfo);

	T **dataT = (T **) data;
	Nd4jLong **shapeInfoPointers = (Nd4jLong **) inputShapeInfos;
	Nd4jLong **tadShapes = (Nd4jLong **) tadPointers;
	Nd4jLong **tadOffsets = (Nd4jLong **) offsetPointers;

	//if (threadIdx.x == 0 && blockIdx.x == 0) {
	//    shape::printShapeInfoLinear("zTadShape", zTadShape);
	//}

    //__shared__ int tDim[1];
    __shared__ int baseIdx;
	__shared__ int yLength;
	__shared__ char yOrder;
	__shared__ int yEWS;
    __shared__ int arrOffset;
	__shared__ int numTads;

	auto zOrder = shape::order(zShapeInfo);
	auto zEWS = shape::elementWiseStride(zShapeInfo);
	auto tadEWS = shape::elementWiseStride(zTadShape);
	auto zLength = shape::length(zShapeInfo);

	if (shape::isVector(zShapeInfo)) {
	
		if (zEWS >= 1) {
			
			for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
				
				if(shape::isVector(shapeInfoPointers[r]) || shape::order(shapeInfoPointers[r]) == shape::order(zShapeInfo)) {
					yLength = shape::length(shapeInfoPointers[r]);
					yEWS = shape::elementWiseStride(shapeInfoPointers[r]);
						// FIXME: this is bad
					__shared__ int baseIdx;
					if (threadIdx.x == 0) {
						baseIdx = 0;
						for (int f = 0; f < r; f++) {
							baseIdx += shape::length(shapeInfoPointers[f]);
						}
					}
					__syncthreads();

					for (int i = threadIdx.x; i < yLength && baseIdx + i < zLength; i += blockDim.x) 
							z[baseIdx + i * zEWS] = dataT[r][i * yEWS];				
					
					__syncthreads();
				} 
				else {
					if (tid == 0)
						printf("Non-matched order for vector\n");
				}
			}
		} 
		else {
			if (tid == 0)
				printf("Vector Non-1 zEWS\n");
		}
		return;
	}

	bool _vec = shape::isVector(zShapeInfo);


	// TODO: to be pulled into separate kernel. matrix concatenation
	for (int r = 0; r < numArrays; r ++) {

		auto currentShape = shapeInfoPointers[r];
		auto currentData = dataT[r];
		auto currentTad = tadShapes[r];
		auto currentOffsets = tadOffsets[r];

		if (threadIdx.x == 0) {

			yLength = shape::length(currentTad);
			yOrder = shape::order(currentTad);
			yEWS = shape::elementWiseStride(currentTad);
            numTads = shape::length(currentShape) / yLength;

            arrOffset = 0;
			for (int f = 0; f < r; f++) 
				arrOffset +=  shape::length(tadShapes[f]);			

		}
		__syncthreads();

        if (yLength == 1 && _vec) {
			
			// edge case, each thread will handle it's own tad then
            for (int j = tid; j < numTads; j += blockDim.x * gridDim.x) {
            	
            	Nd4jLong inputOffset = currentOffsets[j];
				Nd4jLong zOffset = zOffsets[j];

				T *dataTAD = currentData + inputOffset;
				T *zTAD = z + zOffset;

				auto baseOffset = shape::getIndexOffset(arrOffset, zTadShape, shape::length(zTadShape));
				zTAD += baseOffset;

				auto yOffset = shape::getIndexOffset(0, currentTad, shape::length(currentTad));
				zOffset = shape::getIndexOffset(0, zTadShape, shape::length(zTadShape));
				
				zTAD[zOffset] =  dataTAD[yOffset];
            }
		} 
		else {
		
		    for (int j = blockIdx.x; j < numTads; j += gridDim.x) {
				
				auto inputOffset = currentOffsets[j];
				auto zOffset = zOffsets[j];
				auto dataTAD = currentData + inputOffset;
				auto zTAD = z + zOffset;
				    
				auto baseOffset = shape::getIndexOffset(arrOffset, zTadShape, shape::length(zTadShape));				    
				zTAD += baseOffset;

				if (zOrder == yOrder && yEWS > 0  && tadEWS > 0) {
					
					for (int i = threadIdx.x; i < yLength; i += blockDim.x)
						zTAD[i * tadEWS] = dataTAD[i * yEWS];					   
				 } 
				else {
					
					if(tadEWS > 0 && shape::order(zShapeInfo) == shape::order(currentTad)) {
				
						if (threadIdx.x == 0) {
							
							baseIdx = 0;
							for (int f = 0; f < r; f++)
							    	baseIdx += shape::length(shapeInfoPointers[f]);
						}
			    		__syncthreads();

		    			if (numTads == 1) {
	    					for(int k = threadIdx.x; k < yLength; k+= blockDim.x) 
    							zTAD[baseIdx + k * tadEWS] = dataTAD[k];							    
						} 
						else {
							
							for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
								    auto yOffset = shape::getIndexOffset(i, currentTad, yLength);
								    zTAD[baseIdx + i * tadEWS] =  dataTAD[yOffset];
							    }
						}
						    __syncthreads();
					} 
					else {
                    
						for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
							auto yOffset = shape::getIndexOffset(i, currentTad, yLength);
						    auto zOffset = shape::getIndexOffset(i, zTadShape, yLength);
							zTAD[zOffset] =  dataTAD[yOffset];
						}
					}
				}
			    __syncthreads();
			}
		}
		__syncthreads();
	}
}

///////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void execConcatKernel(int dimension,
							int numArrays,
							Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
							void *vz, Nd4jLong *zShapeInfo, 
							Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, 
							Nd4jLong *zTadShape, 
							Nd4jLong *zOffsets) {

	concatKernel<T>(dimension, numArrays, data, inputShapeInfos, vz, zShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}


///////////////////////////////////////////////////////////////////////
template <typename T>
__host__ void concatKernelGeneric(dim3& launchDims, Nd4jPointer* extraPointers,
 							int dimension,
							int numArrays,
							Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
							void *vz, Nd4jLong *zShapeInfo, 
							Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, 
							Nd4jLong *zTadShape, 
							Nd4jLong *zOffsets) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	execConcatKernel<T><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(dimension, numArrays, data, inputShapeInfos, vz, zShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}