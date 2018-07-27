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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_KERNELS_H
#define LIBND4J_SPECIAL_KERNELS_H

#include <helpers/shape.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

/**
* This is utility kernel, that updates given special buffer with proper values in device memory
*/
extern "C" __global__ void prepareShapeBuffer(int *dimension, int *maxDimension, Nd4jLong *specialPointer, int rows) {
    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    dimension[0] = 0;
    maxDimension[0] = 1;

    specialPointer[0] = 2;
    specialPointer[1] = rows;
    specialPointer[2] = 1;
    specialPointer[3] = 1;
    specialPointer[4] = 1;
    specialPointer[5] = 0;
    specialPointer[6] = 1;
    specialPointer[7] = 99;

    //printf("special[0]: [%lld]\n", (long long) specialPointer[0]);
    //shape::printShapeInfoLinear("prepareShapeBuffer", specialPointer);
}

extern "C" __global__ void prepareDimensionalShapeBuffer(Nd4jLong *xShapeInfoBuffer, float *extraParams, Nd4jLong *zShapeInfo) {
    // extraParams[0] - number of dimensions
    // extraParams[1] - dimension
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    int targetDimension = (int) extraParams[1];
    //printf("Target dimension: [%i]\n", targetDimension);

    int targetWidth = shape::shapeOf(xShapeInfoBuffer)[targetDimension];
    //printf("Target rank: [%i]\n", targetWidth);
}

template <typename T>
__device__ void fillIsMaxGeneric(T *dx, long length, long idx) {

   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (long i = tid; i < length; i+= blockDim.x * gridDim.x) {
        dx[i] = (i == idx? 1.0 : 0.0);
   }
}

extern "C" __global__ void fillIsMaxFloat(float *dx, long length, long idx) {
    fillIsMaxGeneric<float>(dx, length, idx);
}

extern "C" __global__ void fillIsMaxDouble(double *dx, long length, long idx) {
    fillIsMaxGeneric<double>(dx, length, idx);
}

extern "C" __global__ void fillIsMaxHalf(float16 *dx, long length, long idx) {
    fillIsMaxGeneric<float16>(dx, length, idx);
}

template <typename T>
__device__ void fillDimensionalIsMaxGeneric(T *dX, Nd4jLong *xShapeInfo, T *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets) {

    __shared__ int tadLength;
    __shared__ int tadEWS;
    __shared__ int numTads;

    __shared__ Nd4jLong *tadShape;
    __shared__ Nd4jLong *tadStride;
    __shared__ int tadRank;
    __shared__ char tadOrder;

    if (threadIdx.x == 0) {
        tadLength = shape::tadLength(zShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(zShapeInfo) / tadLength;

        tadShape = shape::shapeOf(tadOnlyShapeInfo);
        tadStride = shape::stride(tadOnlyShapeInfo);
        tadRank = shape::rank(tadOnlyShapeInfo);
        tadOrder = shape::order(tadOnlyShapeInfo);
    }
    __syncthreads();



    for (int r = blockIdx.x; r < numTads; r+= gridDim.x) {
        auto tadOffsetForBlock = tadOffsets[r];

        int highestElement = (int) dX[r];

        if (dimensionLength > 1 || tadEWS < 1) {
            Nd4jLong xCoord[MAX_RANK];

            for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
                shape::ind2subC(tadRank,tadShape, e, xCoord);

                auto xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                dZ[xOffset] = (e == highestElement? (T) 1.0f : (T) 0.0f);
            }
        } else {
            for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
                // so, we just set dZ[e] for each TAD. Sure, e should be replaced with
                auto idx = tadOffsetForBlock + (e * tadEWS);
                dZ[idx] = (e == highestElement? (T) 1.0f : (T) 0.0f);
            }
        }

    }
}

extern "C" __global__ void fillDimensionalIsMaxFloat(float *dx, Nd4jLong *xShapeInfo, float *dz, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets) {
    fillDimensionalIsMaxGeneric<float>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxDouble(double *dx, Nd4jLong *xShapeInfo, double *dz, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets) {
    fillDimensionalIsMaxGeneric<double>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxHalf(float16 *dx, Nd4jLong *xShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets) {
    fillDimensionalIsMaxGeneric<float16>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

template <typename T>
__device__ void concatKernelGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int zRank = shape::rank(resultShapeInfo);

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

		char zOrder = shape::order(resultShapeInfo);

		int zEWS = shape::elementWiseStride(resultShapeInfo);
		int tadEWS = shape::elementWiseStride(zTadShape);
		int zLength = shape::length(resultShapeInfo);

        __shared__ int arrOffset;
		__shared__ int numTads;


        if (shape::isVector(resultShapeInfo)) {
			//if (threadIdx.x == 0 && blockIdx.x == 0)
			//	printf("Vector here\n");

			if (zEWS >= 1) {
				for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
					if(shape::isVector(shapeInfoPointers[r]) || shape::order(shapeInfoPointers[r]) == shape::order(resultShapeInfo)) {
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
						for (int i = threadIdx.x; i < yLength && baseIdx + i < zLength; i += blockDim.x) {
							result[baseIdx + i * zEWS] = dataT[r][i * yEWS];
						}
						__syncthreads();
					} else {
						if (tid == 0)
							printf("Non-matched order for vector\n");
					}
				}
			} else {
				if (tid == 0)
					printf("Vector Non-1 zEWS\n");
			}
			return;
		}


		bool _vec = shape::isVector(resultShapeInfo);


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
				for (int f = 0; f < r; f++) {
					arrOffset +=  shape::length(tadShapes[f]);
				}

				//if (threadIdx.x == 0 && blockIdx.x == 0) {
			    //    shape::printShapeInfoLinear("currentTad", currentTad);
			    //}
			}
			__syncthreads();

            if (yLength == 1 && _vec) {
				//if (threadIdx.x == 0 && blockIdx.x == 0)
				//	printf("Branch 0\n");

                // edge case, each thread will handle it's own tad then
                for (int j = tid; j < numTads; j += blockDim.x * gridDim.x) {
                    Nd4jLong inputOffset = currentOffsets[j];
				    Nd4jLong resultOffset = zOffsets[j];

				    T *dataTAD = currentData + inputOffset;
				    T *resultTAD = result + resultOffset;

                    Nd4jLong sub[MAX_RANK];

                    if (shape::order(zTadShape) == 'f') {
				        shape::ind2sub(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    } else {
				        shape::ind2subC(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    }
				    Nd4jLong baseOffset = shape::getOffset(0,shape::shapeOf(zTadShape),shape::stride(zTadShape), sub, shape::rank(zTadShape));

				    resultTAD += baseOffset;

					auto yRank = shape::rank(currentTad);
					auto tadRank = shape::rank(zTadShape);

					shape::ind2subC(yRank, shape::shapeOf(currentTad), 0,sub);

					auto yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), sub, yRank);
					resultOffset = shape::getOffset(0, shape::shapeOf(zTadShape), shape::stride(zTadShape), sub, tadRank);

					resultTAD[resultOffset] =  dataTAD[yOffset];
                }
            } else {
				//if (threadIdx.x == 0 && blockIdx.x == 0)
				//	printf("Branch 1\n");

			    for (int j = blockIdx.x; j < numTads; j += gridDim.x) {
				    auto inputOffset = currentOffsets[j];
				    auto resultOffset = zOffsets[j];

				    auto dataTAD = currentData + inputOffset;
				    auto resultTAD = result + resultOffset;

                    Nd4jLong sub[MAX_RANK];

				    shape::ind2subC(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    Nd4jLong baseOffset = shape::getOffset(0,shape::shapeOf(zTadShape),shape::stride(zTadShape), sub, shape::rank(zTadShape));

				    resultTAD += baseOffset;

				    if (zOrder == yOrder && yEWS > 0  && tadEWS > 0) {
				        //if (threadIdx.x == 0 && blockIdx.x == 0)
				        //    printf("Branch A\n");

					    for (int i = threadIdx.x; i < yLength; i += blockDim.x) {
						    resultTAD[i * tadEWS] = dataTAD[i * yEWS];
					    }
				    } else {
					    if(tadEWS > 0 && shape::order(resultShapeInfo) == shape::order(currentTad)) {
					        //if (threadIdx.x == 0 && blockIdx.x == 0)
				            //    printf("Branch B\n");

						    if (threadIdx.x == 0) {
							    baseIdx = 0;
							    for (int f = 0; f < r; f++) {
							    	baseIdx += shape::length(shapeInfoPointers[f]);
						    	}
					    		//printf("R: %i; baseIdx: %i;\n", baseIdx);
				    		}
			    			__syncthreads();

		    				if (numTads == 1) {
	    						for(int k = threadIdx.x; k < yLength; k+= blockDim.x) {
    								resultTAD[baseIdx + k * tadEWS] = dataTAD[k];
							    }
						    } else {
							    Nd4jLong yIdx[MAX_RANK];
							    auto yRank = shape::rank(currentTad);

							    for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
								    shape::ind2subC(yRank, shape::shapeOf(currentTad), i, yIdx);
								    auto yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);

								    resultTAD[baseIdx + i * tadEWS] =  dataTAD[yOffset];
							    }
						    }
						    __syncthreads();
					    } else {
                            //if (threadIdx.x == 0 && blockIdx.x  == 0)
				            //    printf("Branch C; yLength: %i;\n", yLength);

                            Nd4jLong zIdx[MAX_RANK];
						    Nd4jLong yIdx[MAX_RANK];
						    auto yRank = shape::rank(currentTad);
						    auto tadRank = shape::rank(zTadShape);

						    for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
							    shape::ind2subC(yRank, shape::shapeOf(currentTad), i,yIdx);
							    shape::ind2subC(tadRank, shape::shapeOf(zTadShape), i,zIdx);

							    auto yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);
							    auto resultOffset = shape::getOffset(0, shape::shapeOf(zTadShape), shape::stride(zTadShape), zIdx, tadRank);

							    resultTAD[resultOffset] =  dataTAD[yOffset];
						    }
					    }
				    }
				    __syncthreads();
			    }
			}
			__syncthreads();
		}
}

template <typename T>
__device__ void concatKernelScalarGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
    T **input = (T **) data;

    for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x) {
			result[i] = input[i][0];
	}
}

extern "C" __global__ void concatKernelScalarFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelScalarHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelScalarDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


template <typename T>
__device__ void concatKernelHStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    // we expect all data coming in as vectors, and result as 2D matrix
    // the only significant difference here is the fact that input lengths might be different
    auto inputShapes = (Nd4jLong**) inputShapeInfos;
     T **input = (T **) data;

     __shared__ int inputEWS;
     __shared__ int resultEWS;
     __shared__ int inputLength;

     if (threadIdx.x == 0) {
        resultEWS = shape::elementWiseStride(resultShapeInfo);
     }
     __syncthreads();

     for (int r = blockIdx.x; r < numArrays; r+= gridDim.x) {

        __shared__ int baseIdx;
		if (threadIdx.x == 0) {
			baseIdx = 0;
			for (int f = 0; f < r; f++) {
			    baseIdx += shape::length(inputShapes[f]);
		    }
		}
		__syncthreads();


        T *inputData = (T *) input[r];

        if (threadIdx.x == 0) {
         inputEWS = shape::elementWiseStride(inputShapes[r]);
         inputLength = shape::length(inputShapes[r]);
        }
        __syncthreads();

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            result[baseIdx + i * resultEWS] = inputData[i * inputEWS];
        }
        __syncthreads();
     }
}

extern "C" __global__ void concatKernelHStackFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelHStackDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


extern "C" __global__ void concatKernelHStackHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

template <typename T>
__device__ void concatKernelVStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    /*
     this is special case for concat: we group bunch of vectors into 2D matrix
     also: we expect each inputShapeInfo to have EWS, be a vector, and have equal size
     */

     auto inputShapes = (Nd4jLong**) inputShapeInfos;
     T **input = (T **) data;

     __shared__ int inputEWS;
     __shared__ int resultEWS;
     __shared__ int inputLength;

     if (threadIdx.x == 0) {
        inputLength = shape::length(inputShapes[0]);
        inputEWS = shape::elementWiseStride(inputShapes[0]);
        resultEWS = shape::elementWiseStride(resultShapeInfo);
     }
     __syncthreads();

     for (int r = blockIdx.x; r < numArrays; r+= gridDim.x) {

        int resultOffset = r * inputLength * resultEWS;
        T *inputData = (T *) input[r];

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            result[resultOffset + i * resultEWS] = inputData[i * inputEWS];
        }
     }
}

extern "C" __global__ void concatKernelVStackFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelVStackDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelVStackHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


extern "C" __global__ void concatKernelDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets) {
	concatKernelGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}

extern "C" __global__ void concatKernelFloat(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float *result,
											 Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets) {
	concatKernelGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}

extern "C" __global__ void concatKernelHalf(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float16 *result,
											 Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets) {
	concatKernelGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}


template <typename T>
__device__ void pullRowsKernelGeneric(T *x,
                                     Nd4jLong *xShapeInfo,
                                     T *z,
                                     Nd4jLong *zShapeInfo,
                                     Nd4jLong n,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo,
                                     Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo,
                                     Nd4jLong *zTadOffsets) {


    auto xEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(zTadShapeInfo);
    auto tadLength = shape::length(tadShapeInfo);


    if (xEWS >= 1 && zEWS >= 1) {
        for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        }
    } else {

        Nd4jLong xCoord[MAX_RANK];
		Nd4jLong zCoord[MAX_RANK];

        for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                shape::ind2subC(shape::rank(tadShapeInfo),shape::shapeOf(tadShapeInfo), i, xCoord);
		    	shape::ind2subC(shape::rank(zTadShapeInfo),shape::shapeOf(zTadShapeInfo), i, zCoord);

		    	auto xOffset = shape::getOffset(0, shape::shapeOf(tadShapeInfo), shape::stride(tadShapeInfo), xCoord, shape::rank(tadShapeInfo));
	    		auto zOffset = shape::getOffset(0, shape::shapeOf(zTadShapeInfo), shape::stride(zTadShapeInfo), zCoord, shape::rank(zTadShapeInfo));

                rZ[zOffset] = rX[xOffset];
            }
        }
    }
}

extern "C" __global__ void pullRowsKernelHalf(
                                     float16 *x,
                                     Nd4jLong *xShapeInfo,
                                     float16 *z,
                                     Nd4jLong *zShapeInfo,
                                     Nd4jLong n,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo,
                                     Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo,
                                     Nd4jLong *zTadOffsets) {
    pullRowsKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelFloat(float *x,
                                     Nd4jLong *xShapeInfo,
                                     float *z,
                                     Nd4jLong *zShapeInfo,
                                     Nd4jLong n,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo,
                                     Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo,
                                     Nd4jLong *zTadOffsets) {
    pullRowsKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelDouble(double *x,
                                     Nd4jLong *xShapeInfo,
                                     double *z,
                                     Nd4jLong *zShapeInfo,
                                     Nd4jLong n,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo,
                                     Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo,
                                     Nd4jLong *zTadOffsets) {
    pullRowsKernelGeneric<double>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

template <typename T>
__device__ void convertToHalfGeneric(T *dx, Nd4jLong n, half *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = __float2half((float) dx[i]);
    }
}

extern "C" __global__ void kernelFloatsToHalfs(float *dx, Nd4jLong n, half *dz) {
    convertToHalfGeneric<float>(dx, n, dz);
}

extern "C" __global__ void kernelDoublesToHalfs(double *dx, Nd4jLong n, half *dz) {
    convertToHalfGeneric<double>(dx, n, dz);
}

template <typename T>
__device__ void convertHalfsToGeneric(half *dx, Nd4jLong n, T *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = (T) __half2float(dx[i]);
    }
}

extern "C" __global__ void kernelHalfsToDoubles(half *dx, Nd4jLong n, double *dz) {
    convertHalfsToGeneric<double>(dx, n, dz);
}

extern "C" __global__ void kernelHalfsToFloats(half *dx, Nd4jLong n, float *dz) {
    convertHalfsToGeneric<float>(dx, n, dz);
}

/**
 * This kernel accumulates X arrays, and stores result into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template<typename T>
__device__ void accumulateKernelGeneric(T **x, T *z, int n, const Nd4jLong length) {
    __shared__ T *shmem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char sharedmem[];
        shmem = (T *) sharedmem;
    }
    __syncthreads();

    for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
        shmem[threadIdx.x] = 0.0f;

        Nd4jLong baseIdx = r;

        // aggregation step, we roll over all arrays
        for (int ar = 0; ar < n; ar++) {
            T *cdata = (T *) x[ar];
            cdata += baseIdx;

            if (baseIdx + threadIdx.x < length)
                shmem[threadIdx.x] += cdata[threadIdx.x];
        }

        T *wdata = z + baseIdx;

        // saving accumulated values
        if (baseIdx + threadIdx.x < length) {
            wdata[threadIdx.x] = shmem[threadIdx.x];
       }
    }
}


extern "C" __global__ void accumulateKernelHalf(float16 **dx, float16 *dz, int n, Nd4jLong length) {
    accumulateKernelGeneric<float16>(dx, dz, n, length);
}

extern "C" __global__ void accumulateKernelFloat(float **dx, float *dz, int n, Nd4jLong length) {
    accumulateKernelGeneric<float>(dx, dz, n, length);
}

extern "C" __global__ void accumulateKernelDouble(double **dx, double *dz, int n, Nd4jLong length) {
    accumulateKernelGeneric<double>(dx, dz, n, length);
}


template <typename T>
__device__ void averagingKernelGeneric(T **dx, T *dz, int n, Nd4jLong length, bool propagate) {

    __shared__ T *shmem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char sharedmem[];
        shmem = (T *) sharedmem;
    }
    __syncthreads();


    // each block cycles over it's own part of arrays
    for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
        shmem[threadIdx.x] = (T) 0.0f;

        Nd4jLong baseIdx = r;

        // aggregation step, we roll over all arrays
        for (int ar = 0; ar < n; ar++) {
            T *cdata = (T *) dx[ar];
            cdata += baseIdx;

            if (baseIdx + threadIdx.x < length)
                shmem[threadIdx.x] += cdata[threadIdx.x];
        }


        // average data in shared memory
        if (baseIdx + threadIdx.x < length)
            shmem[threadIdx.x] /= n;

        // div step & write out step
        if (dz != nullptr) {
            T *wdata = dz + baseIdx;

            if (baseIdx + threadIdx.x < length) {
                wdata[threadIdx.x] = shmem[threadIdx.x];
            }
        }

        // propagate averaged data to all arrays
        if (propagate)
            for (int ar = 0; ar < n; ar++) {
                T *cdata = (T *) dx[ar];
                cdata += baseIdx;

                if (baseIdx + threadIdx.x < length)
                    cdata[threadIdx.x] = shmem[threadIdx.x];
            }
    }
}


extern "C" __global__ void averagingKernelHalf(float16 **dx, float16 *dz, int n, Nd4jLong length, bool propagate) {
    averagingKernelGeneric<float16>(dx, dz, n, length, propagate);
}

extern "C" __global__ void averagingKernelFloat(float **dx, float *dz, int n, Nd4jLong length, bool propagate) {
    averagingKernelGeneric<float>(dx, dz, n, length, propagate);
}

extern "C" __global__ void averagingKernelDouble(double **dx, double *dz, int n, Nd4jLong length, bool propagate) {
    averagingKernelGeneric<double>(dx, dz, n, length, propagate);
}

template<typename T>
__device__ void tearKernelGeneric(T *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    __shared__ Nd4jLong tadLength;
    __shared__ int tadEWS;
    __shared__ int zEWS;
    __shared__ int tadRank;
    __shared__ Nd4jLong numTads;
    __shared__ int zRank;
    __shared__ Nd4jLong *tadShape;
    __shared__ Nd4jLong *tadStride;
    __shared__ Nd4jLong *zShape;
    __shared__ Nd4jLong *zStride;

    if (threadIdx.x == 0) {
        tadLength = shape::length(tadShapeInfo);
        tadEWS = shape::elementWiseStride(tadShapeInfo);
        zEWS = shape::elementWiseStride(zShapeInfo);
        tadRank = shape::rank(tadShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        zRank = shape::rank(zShapeInfo);
        tadShape = shape::shapeOf(tadShapeInfo);
        tadStride = shape::stride(tadShapeInfo);
        zShape = shape::shapeOf(zShapeInfo);
        zStride = shape::stride(zShapeInfo);
    }
    __syncthreads();

    for (Nd4jLong r = blockIdx.x; r < numTads; r += gridDim.x) {
        T *z = (T *) targets[r];
        T *s = x + tadOffsets[r];

        if (zEWS > 0 && tadEWS > 0) {
        for (Nd4jLong i = threadIdx.x; i < tadLength; i += blockDim.x) {
            z[i * zEWS] = s[i * tadEWS];
        }
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            for (Nd4jLong j = 0; j < tadLength; j++) {
                shape::ind2sub(tadRank,tadShape, j, xCoord);
                shape::ind2sub(zRank, zShape, j, zCoord);

                auto xOffset = shape::getOffset(0, tadShape, tadStride, xCoord, tadRank);
                auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                z[zOffset] = s[xOffset];
            }
        }
    }
}

extern "C" __global__ void tearKernelDouble(double *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearKernelGeneric<double>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

extern "C" __global__ void tearKernelFloat(float *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearKernelGeneric<float>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

extern "C" __global__ void tearKernelHalf(float16 *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearKernelGeneric<float16>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}


template<typename T>
__device__ void shuffleKernelGeneric(T **dX, Nd4jLong **xShapeInfo, T **dZ, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {

            // we assume that shuffle map for each X contains pair TAD Y

            __shared__ int tadLength;
            __shared__ int tadEWS;
            __shared__ int tadRank;
            __shared__ int numTads;
            __shared__ Nd4jLong *tadShape;
            __shared__ Nd4jLong *tadStride;
            __shared__ int yStride;


        for (int f = 0; f < N; f++) {
            T *x = (T *) dX[f];
            T *z = (T *) dZ[f];



            __syncthreads();

            if (threadIdx.x == 0) {
                tadLength = shape::length(tadOnlyShapeInfo[f]);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
                tadRank = shape::rank(tadOnlyShapeInfo[f]);
                numTads = shape::length(xShapeInfo[f]) / tadLength;

                tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
                tadStride = shape::stride(tadOnlyShapeInfo[f]);
            }
            __syncthreads();


            // we roll over the pairs of TADs, thus limit is numTads / 2
            for (Nd4jLong r = blockIdx.x; r < numTads; r += blockDim.x) {
                if (shuffleMap[r] < 0)
                    continue;

                Nd4jLong oldOffset = tadOffsets[f][r];
                Nd4jLong newOffset = tadOffsets[f][shuffleMap[r]];



                T *rX = x + oldOffset;
                T *rY = x + newOffset;

                T *zX = z + oldOffset;
                T *zY = z + newOffset;

                // so we're going to change TAD[oldOffset] with TAD[newOffset]
                if (tadEWS == 1) {
                    for (Nd4jLong i = threadIdx.x; i < tadLength; i += blockDim.x) {
                        T oldX = rX[i];

                        rX[i] = rY[i];
                        zY[i] = oldX;
                    }

                } else {
                    // well have to iterate using ind2sub
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong yCoord[MAX_RANK];

                        for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                            shape::ind2subC(tadRank,tadShape, i, xCoord);
                            shape::ind2subC(tadRank,tadShape, i, yCoord);

                            auto xOffset = shape::getOffset(oldOffset, tadShape, tadStride, xCoord, tadRank);
                            auto yOffset = shape::getOffset(newOffset, tadShape, tadStride, yCoord, tadRank);

                            T oldX = x[xOffset];
                            z[xOffset] = x[yOffset];
                            z[yOffset] = oldX;
                        }
                    }
            }
        }
}


extern "C" __global__ void shuffleKernelDouble(double **x, Nd4jLong **xShapeInfo, double **z, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {
    shuffleKernelGeneric<double>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelFloat(float **x, Nd4jLong **xShapeInfo, float **z, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {
    shuffleKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelHalf(float16 **x, Nd4jLong **xShapeInfo, float16 **z, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {
    shuffleKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}




#endif