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


template <typename T>
__device__ void fillDimensionalIsMaxGeneric(T *dX, Nd4jLong *xShapeInfo, T *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets) {

    __shared__ int tadLength;
    __shared__ int tadEWS;
    __shared__ int numTads;

    if (threadIdx.x == 0) {
        tadLength = shape::tadLength(zShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(zShapeInfo) / tadLength;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < numTads; r+= gridDim.x) {
        auto tadOffsetForBlock = tadOffsets[r];

        int highestElement = (int) dX[r];

        if (dimensionLength > 1 || tadEWS < 1) {

            for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
                
            	auto xOffset = tadOffsetForBlock + shape::getIndexOffset(e, tadOnlyShapeInfo, tadLength);
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


template <typename T>
__device__ void concatKernelGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									void *vresult,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets) {
	
	auto z = static_cast<T*>(vresult);
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
							z[baseIdx + i * zEWS] = dataT[r][i * yEWS];
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
				    Nd4jLong zOffset = zOffsets[j];

				    T *dataTAD = currentData + inputOffset;
				    T *zTAD = z + zOffset;

				    auto baseOffset = shape::getIndexOffset(arrOffset, zTadShape, shape::length(zTadShape));
				    zTAD += baseOffset;

					auto yOffset = shape::getIndexOffset(0, currentTad, shape::length(currentTad));
					zOffset = shape::getIndexOffset(0, zTadShape, shape::length(zTadShape));
					zTAD[zOffset] =  dataTAD[yOffset];
                }
            } else {
				//if (threadIdx.x == 0 && blockIdx.x == 0)
				//	printf("Branch 1\n");

			    for (int j = blockIdx.x; j < numTads; j += gridDim.x) {
				    auto inputOffset = currentOffsets[j];
				    auto zOffset = zOffsets[j];

				    auto dataTAD = currentData + inputOffset;
				    auto zTAD = z + zOffset;
				    
				    auto baseOffset = shape::getIndexOffset(arrOffset, zTadShape, shape::length(zTadShape));				    
				    zTAD += baseOffset;

				    if (zOrder == yOrder && yEWS > 0  && tadEWS > 0) {
				        //if (threadIdx.x == 0 && blockIdx.x == 0)
				        //    printf("Branch A\n");

					    for (int i = threadIdx.x; i < yLength; i += blockDim.x) {
						    zTAD[i * tadEWS] = dataTAD[i * yEWS];
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
    								zTAD[baseIdx + k * tadEWS] = dataTAD[k];
							    }
						    } else {

							    for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
								    auto yOffset = shape::getIndexOffset(i, currentTad, yLength);
								    zTAD[baseIdx + i * tadEWS] =  dataTAD[yOffset];
							    }
						    }
						    __syncthreads();
					    } else {
                            //if (threadIdx.x == 0 && blockIdx.x  == 0)
				            //    printf("Branch C; yLength: %i;\n", yLength);

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

template <typename T>
__device__ void concatKernelScalarGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									void *vresult,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    auto z = static_cast<T*>(vresult);
    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
    T **input = (T **) data;

    for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x) {
			z[i] = input[i][0];
	}
}


template <typename T>
__device__ void concatKernelHStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									void *vresult,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    // we expect all data coming in as vectors, and z as 2D matrix
    // the only significant difference here is the fact that input lengths might be different
    auto z = static_cast<T*>(vresult);
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
            z[baseIdx + i * resultEWS] = inputData[i * inputEWS];
        }
        __syncthreads();
     }
}


template <typename T>
__device__ void concatKernelVStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									void *vresult,
									Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    /*
     this is special case for concat: we group bunch of vectors into 2D matrix
     also: we expect each inputShapeInfo to have EWS, be a vector, and have equal size
     */

	 auto z = static_cast<T*>(vresult);

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

        int zOffset = r * inputLength * resultEWS;
        T *inputData = (T *) input[r];

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            z[zOffset + i * resultEWS] = inputData[i * inputEWS];
        }
     }
}


template <typename T>
__device__ void pullRowsKernelGeneric(void *vx,
                                     Nd4jLong *xShapeInfo,
                                     void *vz,
                                     Nd4jLong *zShapeInfo,
                                     Nd4jLong n,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo,
                                     Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo,
                                     Nd4jLong *zTadOffsets) {

	auto x = static_cast<T*>(vx);
	auto z = static_cast<T*>(vz);
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
        for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
		    	auto xOffset = shape::getIndexOffset(i, tadShapeInfo, tadLength);
		    	auto zOffset = shape::getIndexOffset(i, zTadShapeInfo, tadLength);
                rZ[zOffset] = rX[xOffset];
            }
        }
    }
}



template <typename T>
__device__ void convertToHalfGeneric(T *dx, Nd4jLong n, half *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = __float2half((float) dx[i]);
    }
}

template <typename T>
__device__ void convertHalfsToGeneric(half *dx, Nd4jLong n, T *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = (T) __half2float(dx[i]);
    }
}


/**
 * This kernel accumulates X arrays, and stores z into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template<typename T>
__device__ void accumulateKernelGeneric(void **vx, void *vz, int n, const Nd4jLong length) {

	auto x = reinterpret_cast<T**>(vx);
	auto z = reinterpret_cast<T*>(vz);

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


template <typename T>
__device__ void averagingKernelGeneric(void **vdx, void *vdz, int n, Nd4jLong length, bool propagate) {

	auto dx = reinterpret_cast<T**>(vdx);
	auto dz = reinterpret_cast<T*>(vdz);

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


template<typename T>
__device__ void tearKernelGeneric(void *vx, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto x = static_cast<T*>(vx);

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
        numTads = shape::length(xShapeInfo) / tadLength;
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

            for (Nd4jLong j = 0; j < tadLength; j++) {
                auto xOffset = shape::getIndexOffset(j, tadShapeInfo, tadLength);
                auto zOffset = shape::getIndexOffset(j, zShapeInfo, tadLength);

                z[zOffset] = s[xOffset];
            }
        }
    }
}


template<typename T>
__device__ void shuffleKernelGeneric(void **vdX, Nd4jLong **xShapeInfo, void **vdZ, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {

            // we assume that shuffle map for each X contains pair TAD Y

			auto dX = reinterpret_cast<T**>(vdX);
			auto dZ = reinterpret_cast<T**>(vdZ);

            __shared__ int tadLength;
            __shared__ int tadEWS;
            __shared__ int numTads;

        for (int f = 0; f < N; f++) {
            T *x = (T *) dX[f];
            T *z = (T *) dZ[f];



            __syncthreads();

            if (threadIdx.x == 0) {
                tadLength = shape::length(tadOnlyShapeInfo[f]);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
                numTads = shape::length(xShapeInfo[f]) / tadLength;
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
                        for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {

                            auto xOffset = shape::getIndexOffset(i, tadOnlyShapeInfo[f], tadLength);
                            auto yOffset = newOffset + xOffset;
                            xOffset += oldOffset;

                            T oldX = x[xOffset];
                            z[xOffset] = x[yOffset];
                            z[yOffset] = oldX;
                        }
                    }
            }
        }
}



#endif