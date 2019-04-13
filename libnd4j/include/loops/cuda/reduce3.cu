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

// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.11.2018


#include <op_boilerplate.h>
#include <loops/reduce3.h>
#include <loops/legacy_ops.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace reduce3   {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
__global__ void execScalarGeneric(const int opNum, 
								void *vx, Nd4jLong *xShapeInfo, 
								void *vy, Nd4jLong *yShapeInfo, 
								void *extraParams, 
								void *vz, Nd4jLong *zShapeInfo, 
								int* allocationPointer, 
								void *reductionBuffer,
								Nd4jLong *tadOnlyShapeInfo) {

    Reduce3<X,Z>::execScalarCuda(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, allocationPointer, reductionBuffer, tadOnlyShapeInfo);
}

template <typename X, typename Z>
__global__ void execAllGeneric(const int opNum,
                                      void *vx, Nd4jLong *xShapeInfo,
                                      void *vy, Nd4jLong *yShapeInfo,
                                      void *extraParams,
                                      void *vz, Nd4jLong *zShapeInfo,
                                      int *dimension, int dimensionLength,
                                      int postProcessOrNot,
                                      int *allocationPointer,
                                      Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                                      Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

	Reduce3<X,Z>::execAllCuda(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
__global__ void execGeneric(const int opNum,
								void *vx, Nd4jLong *xShapeInfo,
								void *vy, Nd4jLong *yShapeInfo,
								void *extraParams,
								void *vz, Nd4jLong *zShapeInfo,
								int *dimension, int dimensionLength,
								int postProcessOrNot,
								int *allocationPointer,
								Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
								Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
      
    Reduce3<X,Z>::execCuda(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);    
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__device__ void Reduce3<X,Z>::aggregatePartials(void* vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams) {
	
	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	
	auto sPartials = reinterpret_cast<Z*>(vsPartials);
	auto extraParams = reinterpret_cast<Z *>(vextraParams);
	Nd4jLong floorPow2 = numItems;

	if (floorPow2 & (floorPow2 - 1)) {
		
		while(floorPow2 & (floorPow2 - 1)) 
			floorPow2 &= floorPow2 - 1;
					
		if (tid >= floorPow2) 
			sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
					
		__syncthreads();
	}

	for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
		if (tid < activeThreads) 
			sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
					
		__syncthreads();
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
__device__ void Reduce3<X,Z>::execScalarCuda( void *vx, Nd4jLong *xShapeInfo,
								void *vy, Nd4jLong *yShapeInfo,
								void *extraParams,
								void *vz, Nd4jLong *zShapeInfo, 
								int *allocationPointer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

	auto x = reinterpret_cast<X*>(vx);
	auto y = reinterpret_cast<X*>(vy);
	auto z = reinterpret_cast<Z*>(vz);

	__shared__ Z extraZ[3];
	__shared__ Z* sPartials;
    
	if (threadIdx.x == 0) {
		
		extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<Z*>(shmem);

		extraZ[0] = (Z) 0.0f;
		extraZ[1] = (Z) 0.0f;

		if (extraParams != nullptr)
			extraZ[2] = *(static_cast<Z*>(extraParams));
		else
			extraZ[2] = (Z) 0.0f;
	}

        __syncthreads();

		sPartials[threadIdx.x] = OpType::startingValue(x);
        Nd4jLong length = shape::length(xShapeInfo);
        int xEws = shape::elementWiseStride(xShapeInfo);
        int yEws = shape::elementWiseStride(yShapeInfo);
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        char xOrder = shape::order(xShapeInfo);
        char yOrder = shape::order(yShapeInfo);

        if(xOrder == yOrder && (xEws > 0 && yEws > 0) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {

            if (xEws == 1 && yEws == 1) {
                for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
					sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(x[i], y[i], extraZ), extraZ);
                }
            }
            else {
                for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
					sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(x[i * xEws], y[i * yEws], extraZ), extraZ);
				}
            }
        }
        else {
            sPartials[threadIdx.x] = OpType::startingValue(x);

            for(Nd4jLong i = tid ;i < length; i += gridDim.x * blockDim.x) {
                auto offset  = shape::getIndexOffset(i, xShapeInfo, length);
                auto yOffset = shape::getIndexOffset(i, yShapeInfo, length);
                sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(x[offset], y[yOffset], extraZ), extraZ);
            }
        }

        __syncthreads();
        aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, length), extraZ);
        __syncthreads();

        if (gridDim.x > 1) {

            auto tc = reinterpret_cast<unsigned int *>(reductionBuffer);
            __shared__ bool amLast;
            int rank = shape::rank(xShapeInfo);
            tid = threadIdx.x;
            Z *extraBuffer = (Z *) allocationPointer;
            if (threadIdx.x == 0) {
				reinterpret_cast<Z*>(reductionBuffer)[blockIdx.x] = sPartials[0];
                extraBuffer[blockIdx.x] = extraZ[0];
                extraBuffer[gridDim.x + blockIdx.x] = extraZ[1];
            }

            __threadfence();
            __syncthreads();

            if (threadIdx.x == 0) {
            unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
            amLast = (ticket == gridDim.x - 1);
            }

            sPartials[tid] = OpType::startingValue(x);
            __syncthreads();

            if (amLast) {

                tc[16384] = 0;
                sPartials[threadIdx.x] = OpType::startingValue(x);

                // TODO: later probably replace this. Right now we need extraZ sync for CosineSimilarity ONLY
                if (tid == 0 && extraZ[0] != static_cast<Z>(0) && extraZ[1] != static_cast<Z>(0)) {
                    extraZ[0] = 0.0;
                    extraZ[1] = 0.0;
                    for (int i = 0; i < gridDim.x; i++) {
                        extraZ[0] += extraBuffer[i];
                        extraZ[1] += extraBuffer[gridDim.x + i];
                    }
                }

                for (Nd4jLong i = threadIdx.x; i < gridDim.x; i += blockDim.x)
                        sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], static_cast<Z*>(reductionBuffer)[i], extraZ);

                __syncthreads();
                aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraZ);
                __syncthreads();

                if (threadIdx.x == 0)
                    z[0] = OpType::postProcess(sPartials[0], length, extraZ);
            }
        }
        else {

            if (tid == 0) {
                auto tc = reinterpret_cast<unsigned int*>(reductionBuffer);
                tc[16384] = 0;
                z[0] = OpType::postProcess(sPartials[0], length, extraZ);
            }
        }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
__device__ void Reduce3<X,Z>::transformAll( void *vx, Nd4jLong *xShapeInfo,
											void *vy, Nd4jLong *yShapeInfo,
											void *extraParams,
											void *vz, Nd4jLong *zShapeInfo,
											int *dimension, int dimensionLength,
											int postProcessOrNot,
											int *allocationPointer,
											Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
											Nd4jLong *yTadShapeInfo,Nd4jLong *yOffsets) {

	auto dx = reinterpret_cast<X*>(vx);
	auto dy = reinterpret_cast<X*>(vy);
	auto z = reinterpret_cast<Z*>(vz);

    // initialize partials first
    __shared__ Z* sPartials;
    if(threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<Z*>(shmem);
    }
    __syncthreads();
    
    Z startingVal = OpType::startingValue(dx);
	sPartials[threadIdx.x] = startingVal;
	X *tempX = reinterpret_cast<X*>(sPartials) + blockDim.x;

    const int maxBlock = blockDim.x;

	__shared__ Z extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

    __shared__ int xTadLength;
    __shared__ int yTadLength;

    __shared__ int xTads;
    __shared__ int yTads;                        

    //reading initial data
    if (threadIdx.x == 0) {
		xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

        xTads = shape::length(xShapeInfo) / xTadLength;
        yTads = shape::length(yShapeInfo) / yTadLength;
	}
    __syncthreads();

    int limit = xTadLength / maxBlock;
	if (xTadLength % maxBlock > 0)
		limit++;

	for (int r = blockIdx.x; r < xTads; r += blockDim.x * gridDim.x) {
    	
    	X *x = dx + xOffsets[r];

        if (threadIdx.x < xTadLength && threadIdx.x < maxBlock) {                                
        	auto x0 = shape::getIndexOffset(threadIdx.x, xTadShapeInfo, shape::length(xTadShapeInfo));
            tempX[threadIdx.x] = x[x0];
        }
        __syncthreads();

        for (int g = 0; g < yTads; g++) {
        	
        	X *y = dy + yOffsets[g];
			int ri = (r * yTads) + g;

            sPartials[threadIdx.x] = startingVal;
            if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen)
				extraZ[threadIdx.x] = startingVal;
			__syncthreads();

            // we might have data too large for single cache block, rendering cache useless though :(
            for (int t = 0; t < limit; t++) {

				// we reset tempX IF we have >1 tiles
                if (t >= 1 || (limit > 1 && g > 0))
                	if (threadIdx.x + (t * maxBlock) < xTadLength) {                                            
                    	auto x0 = shape::getIndexOffset(threadIdx.x + (t * maxBlock), xTadShapeInfo, xTadLength);
                    	tempX[threadIdx.x] = x[x0];
                 	}

                    for (int f = threadIdx.x + (t * maxBlock); f < xTadLength && f < threadIdx.x + ((t + 1) * maxBlock); f += blockDim.x * gridDim.x) {                                        
                    	auto y0 = shape::getIndexOffset(f, yTadShapeInfo, yTadLength);
                    	sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(tempX[threadIdx.x], y[y0], extraZ), extraZ);
                    }

                    // we MUST step through this block altogether
					__syncthreads();
       		}

			aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, xTadLength), extraZ);
			__syncthreads();

            if (threadIdx.x == 0) {
                z[ri] = OpType::postProcess(sPartials[threadIdx.x], xTadLength, extraZ);
            }
				
			__syncthreads();
		}
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
__device__ void Reduce3<X,Z>::transform(void *vx, Nd4jLong *xShapeInfo,
										void *vy, Nd4jLong *yShapeInfo,
										void *extraParams,
										void *vz, Nd4jLong *zShapeInfo,
										int *dimension, int dimensionLength,
										int postProcessOrNot, 
										int *allocationPointer,
										Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
										Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
				
	X* x = reinterpret_cast<X*>(vx);
	X* y = reinterpret_cast<X*>(vy);
	Z* z = reinterpret_cast<Z*>(vz);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ int resultScalar;
	__shared__ int xElementWiseStride;
	__shared__ int yElementWiseStride;
	
	//shared memory space for storing intermediate results
	__shared__ Z* sPartials;
    if(threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<Z*>(shmem);
    }
    __syncthreads();
	Z init = OpType::startingValue(x);
	sPartials[threadIdx.x] = init;

	__shared__ Z extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

	//length for the tad
	__shared__ Nd4jLong resultLength;
	__shared__ int tadLength;
	__shared__ int yLength;
	__shared__ int tadEws;
	__shared__ int yTadEws;

	Z startingVal = OpType::startingValue(x);

	Z reduction = OpType::startingValue(x);
	if (threadIdx.x == 0) {
		
		if (zShapeInfo != nullptr)
			resultLength = shape::length(zShapeInfo);
		else 
			resultLength = 1;

		if (dimensionLength == 1) {
			if (dimension == nullptr || dimension[0] == MAX_DIMENSION)
				resultScalar = 1;
			else
				resultScalar = 0;
		}
		else
			resultScalar = 0;

		if (resultLength == 1)
			resultScalar = 1;

		auto xStride = shape::stride(xShapeInfo);
		auto xOrder = shape::order(xShapeInfo);

		tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
		tadEws = shape::elementWiseStride(tadOnlyShapeInfo);
		yLength = shape::length(yShapeInfo);

		if (yTadOnlyShapeInfo != nullptr)
			yTadEws = shape::elementWiseStride(yTadOnlyShapeInfo);
	}
	
	__syncthreads();

    // code branch for TAD vs full array
    if (tadLength == yLength) {                    

		for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
			
			int xOffsetForTad = tadOffsets[i];
			if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) 
				extraZ[threadIdx.x] = startingVal;
				        
			__syncthreads();

			for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
            	auto xOffset = shape::getIndexOffset(j, tadOnlyShapeInfo, tadLength) + xOffsetForTad;
                auto yOffset = shape::getIndexOffset(j, yShapeInfo, tadLength);
				sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(x[xOffset],y[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(x[xOffset],y[yOffset], extraZ), extraZ);
			}
			
			__syncthreads();
			aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);
            __syncthreads();
            
            if (threadIdx.x == 0)
				z[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);
			__syncthreads();
		}
	} 
	else if (!resultScalar) {
		
		if(tadEws >= 1 && yTadEws) {
    	
    		for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
				int xOffsetForTad = tadOffsets[i];
				int yOffsetForTad = yTadOffsets[i];

				if(OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) 
					extraZ[threadIdx.x] = startingVal;
				            
				__syncthreads();

            	if (threadIdx.x < tadLength)
					sPartials[threadIdx.x] =  OpType::op(x[xOffsetForTad + tadEws * threadIdx.x],y[yOffsetForTad + yTadEws * threadIdx.x], extraZ);

				for(int j = threadIdx.x + blockDim.x; j < tadLength; j += blockDim.x) 
					sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], OpType::op(x[xOffsetForTad + tadEws * j],y[yOffsetForTad + yTadEws * j], extraZ), extraZ);
					
				__syncthreads();
				aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);
				__syncthreads();
                
                if (threadIdx.x == 0)
					z[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);
				__syncthreads();
			}
		}
		else {
/*
						// DO NOT REMOVE THIS COMMENTED BLOCK PLEASE

						for (int r = blockIdx.x; r < tad->numTads; r += gridDim.x) {
                            if (threadIdx.x == 0)
                                tad->createOffsetForBlock(r);
                            __syncthreads();

                            int tadOffsetForBlock = tad->tadOffsetForBlock;
                            X *xVal = x + tadOffsetForBlock;


                            sPartials[threadIdx.x] = this->startingValue(xVal);
                            for(int i = threadIdx.x; i < tad->tadLength; i+= blockDim.x) {
                    			int xOffsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength, nullptr);
								int yOffsetForTad = shape::tadOffset(i, yShapeInfo, dimension, dimensionLength, nullptr);

                                sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],x[tadOffsetForBlock + i *  tad->tadEws], extraParams);
                            }
                            __syncthreads();

                            // aggregate. do NOT reduce for elements > tadLength
                            aggregatePartials(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength), extraParams);


                            __syncthreads();
                            if (threadIdx.x == 0)
                                z[r] = this->postProcess(sPartials[threadIdx.x], tad->tadLength, extraParams);
                        }

*/
			for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
				
				auto xOffsetForTad = tadOffsets[i];
				auto yOffsetForTad = yTadOffsets[i];

					if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) 
						extraZ[threadIdx.x] = startingVal;
				    
				    __syncthreads();

					for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                    	auto xOffset = xOffsetForTad + shape::getIndexOffset(j, tadOnlyShapeInfo,  tadLength);
                        auto yOffset = yOffsetForTad + shape::getIndexOffset(j, yTadOnlyShapeInfo, tadLength);
						sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(x[xOffset],y[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(x[xOffset],y[yOffset], extraZ), extraZ);
					}
					
					__syncthreads();                    
					aggregatePartials<OpType>(reinterpret_cast<void*>(sPartials), threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);
                    __syncthreads();
                    	
                    if (threadIdx.x == 0)
						z[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

					__syncthreads();
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__device__ void Reduce3<X,Y>::execCuda(const int opNum,
									void *vx, Nd4jLong *xShapeInfo,
									void *vy, Nd4jLong *yShapeInfo,
									void *extraParams,
									void *vz, Nd4jLong *zShapeInfo,
									int *dimension, int dimensionLength,
									int postProcessOrNot,
									int *allocationPointer,
									Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
									Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
                           
	DISPATCH_BY_OPNUM_TT(transform, PARAMS(vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
}



//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__device__ void Reduce3<X,Y>::execAllCuda( const int opNum,
										void *vx, Nd4jLong *xShapeInfo,
										void *vy, Nd4jLong *yShapeInfo,
										void *extraParams,
										void *vz, Nd4jLong *zShapeInfo,
										int *dimension, int dimensionLength,
										int postProcessOrNot,
										int *allocationPointer,
										Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
										Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

	DISPATCH_BY_OPNUM_TT(transformAll, PARAMS(vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__device__ void Reduce3<X,Y>::execScalarCuda(const int opNum,
										void *vx, Nd4jLong *xShapeInfo,
										void *vy, Nd4jLong *yShapeInfo,
										void *extraParams,
										void *vz, Nd4jLong *zShapeInfo,
										int * allocationPointer, void *reductionBuffer, 										
										Nd4jLong *tadOnlyShapeInfo) {

	DISPATCH_BY_OPNUM_TT(execScalarCuda, PARAMS(vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, allocationPointer, reductionBuffer, tadOnlyShapeInfo), REDUCE3_OPS);
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
__host__ void Reduce3<X,Z>::exec(dim3 launchDims, cudaStream_t *stream, 
									int opNum,
									void *vx, Nd4jLong *xShapeInfo,
									void *vy, Nd4jLong *yShapeInfo,		
									void *extraParams,
									void *vz, Nd4jLong *zShapeInfo,
									int *dimension, int dimensionLength,		
									int postProcessOrNot, 
									int *allocationPointer, 
									Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, 
									Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
        
    execGeneric<X, Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
}

////////////////////////////////////////////////////////////////////////
	template <typename X, typename Z>
	__host__ void Reduce3<X,Z>::execAll(dim3 launchDims, cudaStream_t *stream,
									 int opNum,
									 void *vx, Nd4jLong *xShapeInfo,
									 void *vy, Nd4jLong *yShapeInfo,
									 void *extraParams,
									 void *vz, Nd4jLong *zShapeInfo,
									 int *dimension, int dimensionLength,
									 int postProcessOrNot,
									 int *allocationPointer,
									 Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
									 Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

		execAllGeneric<X, Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
	}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
__host__ void Reduce3<X,Z>::execScalar(dim3 launchDims, cudaStream_t *stream, 
										int opNum, 
										void *vx, Nd4jLong *xShapeInfo, 
										void *vy, Nd4jLong *yShapeInfo, 
										void *extraParams, 
										void *vz, Nd4jLong *zShapeInfo, 
										int* allocationPointer, 
										void *reductionBuffer,
										Nd4jLong *tadOnlyShapeInfo) {
        
    execScalarGeneric<X,Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(opNum, vx, xShapeInfo, vy, yShapeInfo, extraParams, vz, zShapeInfo, allocationPointer, reductionBuffer, tadOnlyShapeInfo);
}





	BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT Reduce3, , LIBND4J_TYPES, FLOAT_TYPES);

}
}