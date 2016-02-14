/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#define EXTRA_PARAMS_LENGTH 10

#include <op.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <sharedmem.h>
#include <omp.h>
#ifdef __JNI__
#include <jni.h>
#endif

namespace functions {
    namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
        template<typename T>
        class Reduce3: public virtual functions::ops::Op<T> {

        public:

            virtual
#ifdef __CUDACC__
            __host__  __device__

#endif
            inline T postProcess(T reduction, int n,T **extraParamsRef) = 0;

            virtual
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            T startingValue(T *input) = 0;

            virtual
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            T * generateExtraParams() = 0;
            virtual
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            void finalizeExtraParams(T **extraParamsRef)  = 0;

            /**
             *
             * @param d1
             * @param d2
             * @param extraParams
             * @return
             */
            //an op for the kernel
            virtual
#ifdef __CUDACC__
            __host__  __device__

#endif
            inline T op(T d1, T d2, T **extraParamsRef) = 0;

            //calculate an update of the reduce operation
            /**
             *
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
            virtual
#ifdef __CUDACC__
            __host__  __device__

#endif
            inline T update(T old, T opOutput, T **extraParamsRef) = 0;

            /**
             *
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
            virtual
#ifdef __CUDACC__
            __host__  __device__

#endif
            inline T merge(T old, T opOutput, T **extraParamsRef) = 0;




            /**
             *
             * @param d1
             * @param d2
             * @param extraParams
             * @return
             */
            //an op for the kernel
#ifdef __CUDACC__
            virtual __device__

	inline T opAtomic(T d1, T d2, T **extraParamsRef) = 0;
#endif

#ifdef __CUDACC__
            /**
	 * Aggregate shared memory
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __inline__ __device__ void aggregatePartials(T **sPartialsRef, int tid, T **extraParamsRef) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		T *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
			while (floorPow2 & (floorPow2 - 1)) {
				floorPow2 &= floorPow2 - 1;
			}
			if (tid >= floorPow2) {
				sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2], sPartials[tid], extraParamsRef);
			}
			__syncthreads();
		}

		for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads) {
				sPartials[tid] = update(sPartials[tid], sPartials[tid + activeThreads], extraParamsRef);
			}
			__syncthreads();
		}
	}

	/**

	 Perform a reduction
	 @param n the number of elements
	 @param xOffset the starting offset
	 @param dx the data to perform the reduction on
	 @param incx the increment on which to perform the reduction
	 @param extraParams extra parameters used for calculations
	 @param result where to store the result of the reduction
	 */
	virtual __inline__ __device__ void transform(
			int n, T *dx, int *xShapeInfo,
			T *dy,
			int *yShapeInfo, T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *gpuInformation,
			int *dimension,
			int dimensionLength, int postProcessOrNot) {
		/**
		 * Gpu information for the problem
		 */
		int tid = threadIdx.x;

		__shared__ volatile int resultScalar;

		__shared__ int xElementWiseStride;
		__shared__ int xOffset;

		__shared__ int yElementWiseStride;
		__shared__ int yOffset;

		//shared memory space for storing intermediate results
		SharedMemory <T> val;
		volatile T *sPartials = val.getPointer();
		T startingVal = this->startingValue(dx);


		int numElements = gpuInformation[2] / sizeof(T);
		for (int i = tid; i < numElements; i += blockDim.x)
			sPartials[i] = startingVal;
		__syncthreads();



		//length for the tad
		__shared__ int reductionIndexesPerBlock;
		__shared__ int tensorsForDimension;

		//starting index for tad
		__shared__ volatile int currentBlockOffset;
		//ending index for tad
		__shared__ volatile int endingOffset;
		//length for the tad
		__shared__ volatile int xLength;

		__shared__ volatile int resultLength;


		__shared__ int elementsPerTad;

		//only compute the tad indexes once
		__shared__ shape::TADPermuteInfo xTadInfo;
		__syncthreads();

		__shared__ shape::TADPermuteInfo yTadInfo;
		__syncthreads();

		__shared__ T *newExtraParams;


		T reduction = this->startingValue(dx);
		if (tid == 0) {
			if (dimensionLength == 1) {
				if (dimension[0] == shape::MAX_DIMENSION)
					resultScalar = 1;
				else
					resultScalar = 0;
			}
			else
				resultScalar = 0;
			tensorsForDimension = shape::tensorsAlongDimension(xShapeInfo, dimension, dimensionLength);

			xOffset = shape::offset(xShapeInfo);
			xElementWiseStride = shape::elementWiseStride(xShapeInfo);
			xLength = shape::length(xShapeInfo);


			resultLength = shape::prod(shape::shapeOf(resultShapeInfo), shape::rank(resultShapeInfo));
			xOffset = shape::offset(xShapeInfo);
			xElementWiseStride = shape::elementWiseStride(xShapeInfo);
			elementsPerTad = xLength / resultLength;

			yElementWiseStride = shape::elementWiseStride(yShapeInfo);
			yOffset = shape::offset(yShapeInfo);
			if (gridDim.x >= resultLength) {
				reductionIndexesPerBlock = 1;
			}
			else {
				reductionIndexesPerBlock = resultLength / gridDim.x;
			}
		}

		__syncthreads();

		T curr, currY;

		if(!resultScalar && dimensionLength > 1) {
			int resultLength = shape::length(resultShapeInfo);

			if(tid == 0) {
				xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
				yTadInfo = shape::tadInfo(yShapeInfo, dimension, dimensionLength);

			}
			__syncthreads();



			if(tid >= resultLength)
				return;


			/**
			 * The element wise stride belongs to a reduction index.
			 * When used out of order, we can get rid of the data
			 * dependencies and rely on using the max dimension
			 * specified for stride instead.
			 * Say we take the sum(0,1) along arr
			 * we can use arr.stride(1) as a representation
			 * along which to iterate.
			 */


			//local view with pointer arithmetic
			T *localExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength()) : NULL;
			int tadElementWiseStride = shape::stride(xShapeInfo)[dimensionLength - 1];
			int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
			int tadLength = xTadInfo.tensorShapeProd;
			int xLength = shape::length(xShapeInfo);

#pragma unroll
			for(int i = tid; i < resultLength; i+= gridDim.x * blockDim.x) {
				if(this->extraParamsLength() > 0) {
					for(int k = 0; k < this->extraParamsLength(); k++) {
						localExtraParams[k] = this->startingValue(dx);
					}
				}

				int offset = dimensionLength > 1 ? i : tadLength * i;
				sPartials[tid] = op(dx[offset], dy[offset],&localExtraParams);
				__syncthreads();
				for(int j = 1; j < elementsPerReductionIndex; j++) {
					sPartials[tid] =  update(sPartials[tid],this->op(dx[offset + tadElementWiseStride * j],dy[offset + tadElementWiseStride * j], &localExtraParams), &localExtraParams);
					__syncthreads();

				}


				result[i] = postProcess(sPartials[tid],tadLength,&localExtraParams);
			}


			if(this->extraParamsLength() > 0)
				free(localExtraParams);

			if(tid == 0) {
				shape::freePermuteInfo(xTadInfo);
			}

		}

		else if (resultScalar) {
			if(blockIdx.x >= resultLength)
				return;
			unsigned int i = xOffset + blockIdx.x * xElementWiseStride + tid;
			unsigned int j = yOffset + blockIdx.x * yElementWiseStride + tid;
			unsigned int gridSize = blockDim.x * gridDim.x * xElementWiseStride;
			unsigned int gridSizeY = blockDim.x * gridDim.x * yElementWiseStride;
			if(tid == 0) {
				newExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength() * resultLength) : extraParams;
				for(int i = 0; i < this->extraParamsLength(); i++) {
					newExtraParams[i] = this->startingValue(dx);
				}
			}

			__syncthreads();

			if(xOffset == 0 && yOffset == 0 && xElementWiseStride == 1 && yElementWiseStride == 1) {
#pragma unroll
				while (i  < n && j  < n) {
					curr = dx[i];
					currY = dy[j];

					/**
					 * Find why extra params vals
					 * aren't getting updated properly.
					 *
					 */
					reduction = update(reduction, this->opAtomic(curr, currY, &newExtraParams), &newExtraParams);
					__syncthreads();
					i += gridSize;
					j += gridSizeY;
				}

			}
			else {
				// we reduce multiple elements per thread.  The number is determined by the
				// number of active thread blocks (via gridDim).  More blocks will result
				// in a larger gridSize and therefore fewer elements per thread
#pragma unroll
				while (i * xElementWiseStride < n && j * yElementWiseStride < n) {
					curr = dx[i];
					currY = dy[j];
					reduction = update(reduction, this->opAtomic(curr, currY, &newExtraParams), &newExtraParams);
					__syncthreads();
					i += gridSize;
					j += gridSizeY;
				}

			}


			// each thread puts its local sum into shared memory
			sPartials[tid] = reduction;
			__syncthreads();

			T **sPartialsRef = (T **) &sPartials;
			aggregatePartials(sPartialsRef, tid, &newExtraParams);
			/**
			 * Look at something that uses the extra params
			 * and aggregates the extra values propelry.
			 *This will be used in summary stats too.
			 */
			// write result for this block to global mem
			if (tid == 0) {
				if (postProcessOrNot) {
					result[blockIdx.x] = postProcess(sPartials[0], xLength,&newExtraParams);
				}
				else {
					result[blockIdx.x] = sPartials[0];
				}


			}


			if(tid == 0 && this->extraParamsLength() > 0)
				this->finalizeExtraParams(&newExtraParams);

		}

		else if (!resultScalar) {
			__shared__ int *tadShapeBuffer;
			if(tid == 0) {
				xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
				yTadInfo = shape::tadInfo(yShapeInfo, dimension, dimensionLength);
				tadShapeBuffer = shape::shapeBuffer(xTadInfo.tensorShapeLength,xTadInfo.tensorShape);
				newExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength() * resultLength) : extraParams;
				for(int i = 0; i < this->extraParamsLength(); i++) {
					newExtraParams[i] = this->startingValue(dx);
				}
			}
			__syncthreads();

			if (reductionIndexesPerBlock * blockIdx.x >= resultLength)
				return;

			int tadsPerReductionIndex = tensorsForDimension / resultLength;

			//minimum number of threads needed for each reduction index
			int tadsNeeded = reductionIndexesPerBlock * tadsPerReductionIndex;

			//don't need all threads
			if (tid >= tadsNeeded)
				return;
			else {
				//process each tad
				//tad wrt the thread
				int currTad = tid + (blockIdx.x * reductionIndexesPerBlock);
				int offsetForTad = shape::offset(currTad, xShapeInfo, dimension,dimensionLength, xTadInfo);
				int yOffsetForTad = shape::offset(currTad, yShapeInfo, dimension,dimensionLength, yTadInfo);
				if(xElementWiseStride > 1 && yElementWiseStride > 1) {
					//update the reduction for the thread for the current tad
					//note here that we compute the offset and then accumulate in shared memory
#pragma unroll
					for (int element = 0;
							element < elementsPerTad; element++, offsetForTad += xElementWiseStride,yOffsetForTad += yElementWiseStride) {
						sPartials[tid] = update(sPartials[tid], op(dx[offsetForTad],dy[yOffsetForTad],&newExtraParams), &newExtraParams);
						__syncthreads();
					}
				}
				else {
					//update the reduction for the thread for the current tad
					//note here that we compute the offset and then accumulate in shared memory
					for (int element = 0;
							element < elementsPerTad; element++, offsetForTad += xElementWiseStride,yOffsetForTad += yElementWiseStride) {
						sPartials[tid] = update(sPartials[tid], op(dx[offsetForTad],dy[yOffsetForTad],&newExtraParams), &newExtraParams);
						__syncthreads();
					}
				}


			}

			//first thread for a reduction index
			if (tid % tadsPerReductionIndex == 0 && tadsPerReductionIndex > 1) {
				/**
				 * Each reduction index is handled by k tads
				 * which need to be combined in each thread.
				 *
				 * Since the TADS to be combined
				 * are to be next to each other
				 * we can assume that
				 * the items in shared memory
				 * can be combined and collapsed
				 * in to the first thread's
				 * entry.
				 *
				 * This follows a similar pattern
				 * for global block wise reduction
				 * and computing parallel sums
				 * in other reduction implementations.
				 *
				 */
#pragma unroll
				for (int i = 1; i < tadsPerReductionIndex; i++) {
					sPartials[tid] = update(sPartials[tid], sPartials[tid + i], &newExtraParams);
					__syncthreads();
				}
			}

			__syncthreads();

			//after all the threads are done processing each first item in shared memory
			//should correspond to the final value for the particular reduction index
			//that was set for this block.
			if (tid == 0) {
#pragma unroll
				for (int i = 0; i < reductionIndexesPerBlock; i++) {
					int reductionIndexToProcess = i + blockIdx.x * reductionIndexesPerBlock;
					if (postProcessOrNot) {
						result[reductionIndexToProcess] = postProcess(sPartials[i], xLength,&newExtraParams);
					}
					else {
						result[reductionIndexToProcess] = sPartials[i];
					}
				}

				free(tadShapeBuffer);
				shape::freePermuteInfo(xTadInfo);
				shape::freePermuteInfo(yTadInfo);

			}

			if(tid == 0 && this->extraParamsLength() > 0)
				this->finalizeExtraParams(&newExtraParams);

		}



	}





#endif

            /**
             *
             * @param x
             * @param xShapeInfo
             * @param extraParamsVals
             * @param y
             * @param yShapeInfo
             * @param result
             * @param resultShapeInfo
             */
#ifdef __CUDACC__
            __host__
#endif
            T execScalar(
                    T *x,
                    int *xShapeInfo,
                    T *extraParamsVals,
                    T *y, int *yShapeInfo) {


                T startingVal = this->startingValue(x);
                int length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
                for(int i = 0; i < this->extraParamsLength();i++) {
                    extraParamsVals[i] = startingVal;
                }


                if (xElementWiseStride == 1) {
                    int i;
#pragma omp parallel private(i)
                    {
                        T local = this->startingValue(x);
                        T *localExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength()) : NULL;
                        for(int extraParamsIdx = 0; extraParamsIdx < this->extraParamsLength(); extraParamsIdx++) {
                            localExtraParams[extraParamsIdx] = startingVal;
                        }

#pragma omp simd
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            local = update(local, op(x[i], y[i], &localExtraParams),
                                           &(localExtraParams));
                        }

#pragma omp critical
                        {

                            if(this->extraParamsLength() > 0)
                                this->aggregateExtraParams(&extraParamsVals,&localExtraParams);
                            startingVal = update(local,startingVal,&extraParamsVals);
                        }

                        if(this->extraParamsLength() > 0)
                            free(localExtraParams);

                    }

                    return postProcess(startingVal, length,&(extraParamsVals));

                }

                else {
                    int i;
#pragma omp parallel private(i)
                    {
                        T local = this->startingValue(x);
                        T *localExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength()) : NULL;
#pragma omp simd
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            local = update(local,
                                           op(x[i * xElementWiseStride], y[i * yElementWiseStride],
                                              &localExtraParams), &(localExtraParams));
                        }

#pragma omp critical
                        {
                            if(this->extraParamsLength() > 0)
                                this->aggregateExtraParams(&extraParamsVals,&localExtraParams);
                            startingVal = update(local,startingVal,&extraParamsVals);
                        }

                        if(this->extraParamsLength() > 0)
                            free(localExtraParams);
                    }


                    return  postProcess(startingVal, length,&(extraParamsVals));

                }

            }
            /**
             *
             * @param x
             * @param xShapeInfo
             * @param extraParamsVals
             * @param y
             * @param yShapeInfo
             * @param result
             * @param resultShapeInfo
             */
#ifdef __CUDACC__
            __host__
#endif
            void exec(
                    T *x,
                    int *xShapeInfo,
                    T *extraParamsVals,
                    T *y, int *yShapeInfo,
                    T *result, int *resultShapeInfo) {

               result[0] = execScalar(x,xShapeInfo,extraParamsVals,y,yShapeInfo);
            }

            /**
             *
             * @param x
             * @param xShapeInfo
             * @param extraParamsVals
             * @param y
             * @param yShapeInfo
             * @param result
             * @param resultShapeInfoBuffer
             * @param dimension
             * @param dimensionLength
             */
            void exec(T *x, int *xShapeInfo,
                      T *extraParamsVals,
                      T *y, int *yShapeInfo,
                      T *result,
                      int *resultShapeInfoBuffer,
                      int *dimension,
                      int dimensionLength) {
                if(shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar(x,xShapeInfo,extraParamsVals,y,yShapeInfo);
                    return;
                }

                T startingVal = this->startingValue(x);

                shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo,dimension, dimensionLength);
                int resultLength = shape::length(resultShapeInfoBuffer);
                /**
                 * The element wise stride belongs to a reduction index.
                 * When used out of order, we can get rid of the data
                 * dependencies and rely on using the max dimension
                 * specified for stride instead.
                 * Say we take the sum(0,1) along arr
                 * we can use arr.stride(1) as a representation
                 * along which to iterate.
                 */
                int tadElementWiseStride = dimensionLength > 1 ? shape::stride(xShapeInfo)[dimensionLength - 1] : shape::computeElementWiseStride(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),shape::order(xShapeInfo) == 'f',dimension,dimensionLength);
                int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
                int tadLength = tadPermuteInfo.tensorShapeProd;
#pragma omp parallel for
                    for(int i = 0; i < resultLength; i++) {
                        T *localExtraParams = this->extraParamsLength() > 0 ? (T *) malloc(sizeof(T) * this->extraParamsLength()) : NULL;
                        for(int extraParamsIdx = 0; extraParamsIdx < this->extraParamsLength(); extraParamsIdx++) {
                            localExtraParams[extraParamsIdx] = startingVal;
                        }

                        int offset = dimensionLength > 1 ? i : shape::offset(i,xShapeInfo,dimension,dimensionLength,tadPermuteInfo);
                        result[i] = op(x[offset], y[offset],&localExtraParams);
                        for(int j = 1; j < elementsPerReductionIndex; j++) {
                            result[i] =  update(result[i],op(x[offset + tadElementWiseStride * j],y[offset + tadElementWiseStride * j], &localExtraParams), &localExtraParams);
                        }

                        result[i] = postProcess(result[i],tadLength,&localExtraParams);

                        if(localExtraParams != NULL)
                            free(localExtraParams);
                    }





                shape::freePermuteInfo(tadPermuteInfo);
            }



#ifdef __CUDACC__
            __host__ __device__
#endif
            virtual ~Reduce3() {
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            Reduce3() {
            }

        };

        namespace ops {
/**
 * Cosine similarity between 2
 * arrays
 */
            template<typename T>
            class CosineSimilarity: public virtual Reduce3<T> {
            public:

                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T * generateExtraParams() {
                    T *extraParams = (T *) malloc(sizeof(T) * 2);
                    return extraParams;
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                void finalizeExtraParams(T **extraParams)  {
                    free(*extraParams);
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T startingValue(T *input) {
                    return 0.0;
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                inline T postProcess(T reduction, int n,T **extraParamsRef) {
                    T *extraParams = *extraParamsRef;
                    return reduction / (nd4j::math::nd4j_sqrt<T>(extraParams[0]) * nd4j::math::nd4j_sqrt<T>(extraParams[1]));
                }
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T **extraParamsRef) {
                    T *extraParams = *extraParamsRef;
                    extraParams[0] += d1 * d1;
                    extraParams[1] += d2 * d2;
                    return (d1 * d2);
                }


#ifdef __CUDACC__
                __host__ __device__
#endif
                void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                    T *extraParamsTotalRef = *extraParamsTotal;
                    T *extraParamsLocalRef = *extraParamsLocal;
                    extraParamsTotalRef[0] += extraParamsLocalRef[0];
                    extraParamsTotalRef[1] += extraParamsLocalRef[1];

                }


                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
#ifdef __CUDACC__
                virtual __device__
	inline T opAtomic(T d1, T d2, T **extraParamsRef) {
		T *extraParams = *extraParamsRef;

		nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],d1 * d1);
		nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],d2 * d2);

		return (d1 * d2);
	}
#endif
                //calculate an update of the reduce operation
                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T update(T old, T opOutput, T **extraParamsRef) {
                    return old + opOutput;
                }

                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T merge(T old, T opOutput, T **extraParamsRef) {
                    return update(old, opOutput, extraParamsRef);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("cosinesimilarity_strided");
                }




#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual ~CosineSimilarity() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                CosineSimilarity() {
                    this->extraParamsLen = 2;
                }
            };


/**
 * Dot product between 2 arrays
 */
            template<typename T>
            class Dot: public virtual Reduce3<T> {
            public:
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T * generateExtraParams() {
                    return NULL;
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                void finalizeExtraParams(T **extraParamsRef)  {
                    //no-op
                    free(*extraParamsRef);
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T startingValue(T *input) {
                    return 0.0;
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                inline T postProcess(T reduction, int n,T **extraParamsRef) {
                    return reduction;
                }
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T **extraParamsRef) {
                    return d1 * d2;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel

#ifdef __CUDACC__
                virtual
	__device__


	inline T opAtomic(T d1, T d2, T **extraParamsRef) {
		return op(d1,d2,extraParamsRef);
	}
#endif

                //calculate an update of the reduce operation
                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T update(T old, T opOutput, T **extraParamsRef) {
                    return opOutput + old;
                }

                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T merge(T old, T opOutput, T **extraParamsRef) {
                    return update(old, opOutput, extraParamsRef);
                }

#ifdef __CUDACC__
                __host__ __device__
#endif
                void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                    //no extra params aggregation needs to happen
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("euclidean_strided");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual ~Dot() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Dot() {
                }
            };



/**
 * Euclidean distance between 2 arrays
 */
            template<typename T>
            class EuclideanDistance: public virtual Reduce3<T> {
            public:
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T * generateExtraParams() {
                    return NULL;
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                void finalizeExtraParams(T **extraParamsRef)  {
                    //no-op
                    free(*extraParamsRef);
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T startingValue(T *input) {
                    return 0.0;
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                inline T postProcess(T reduction, int n,T **extraParamsRef) {
                    return nd4j::math::nd4j_sqrt<T>(reduction);
                }
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T **extraParamsRef) {
                    T ret = d1 - d2;
                    return ret * ret;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel

#ifdef __CUDACC__
                virtual
	__device__


	inline T opAtomic(T d1, T d2, T **extraParamsRef) {
		return op(d1,d2,extraParamsRef);
	}
#endif

                //calculate an update of the reduce operation
                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T update(T old, T opOutput, T **extraParamsRef) {
                    return opOutput + old;
                }

                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T merge(T old, T opOutput, T **extraParamsRef) {
                    return update(old, opOutput, extraParamsRef);
                }

#ifdef __CUDACC__
                __host__ __device__
#endif
                void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                    //no extra params aggregation needs to happen
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("euclidean_strided");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual ~EuclideanDistance() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                EuclideanDistance() {
                }
            };


/**
 * Manhattan distance between 2 arrays
 */
            template<typename T>
            class ManhattanDistance: public virtual Reduce3<T> {
            public:
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T * generateExtraParams() {
                    return NULL;
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                void finalizeExtraParams(T **extraParamsRef)  {
                    //no op
                    free(*extraParamsRef);
                }
                virtual
#ifdef __CUDACC__
                __inline__ __host__ __device__
#endif
                T startingValue(T *input) {
                    return 0.0;
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                inline T postProcess(T reduction, int n,T **extraParamsRef) {
                    return reduction;
                }
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T **extraParamsRef) {
                    return nd4j::math::nd4j_abs<T>(d1 - d2);
                }

                //calculate an update of the reduce operation
                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T update(T old, T opOutput, T **extraParamsRef) {
                    return old + opOutput;
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                    //no extra params aggregation needs to happen
                }
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel

#ifdef __CUDACC__
                virtual	__device__


	inline T opAtomic(T d1, T d2, T **extraParamsRef) {
		return op(d1,d2,extraParamsRef);
	}
#endif

                /**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T merge(T old, T opOutput, T **extraParamsRef) {
                    return update(old, opOutput, extraParamsRef);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("manhattan_strided");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual ~ManhattanDistance() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                ManhattanDistance() {
                }
            };

        }

        template<typename T>
        class Reduce3OpFactory {
        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            Reduce3OpFactory() {
            }


            /**
             * Create an op given an op number
             * @param op the op number
             * 0: manhattan distance
             * 1: euclidean distance
             * 2: cosine similarity
             * @return
             */
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            Reduce3<T> * getOp(int op) {
                if (op == 0)
                    return new functions::reduce3::ops::ManhattanDistance<T>();
                else if (op == 1)
                    return new functions::reduce3::ops::EuclideanDistance<T>();
                else if (op == 2)
                    return new functions::reduce3::ops::CosineSimilarity<T>();
                else if (op == 3)
                    return new functions::reduce3::ops::Dot<T>();
                return NULL;
            }
        };

    }
}

#ifdef __CUDACC__

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
template <typename T>
__device__ void reduce3Generic(
		int opNum,
		int n, T *dx, int *xShapeInfo,
		T *dy,
		int *yShapeInfo, T *extraParams, T *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	__shared__ functions::reduce3::Reduce3<T> * op;
	__shared__ functions::reduce3::Reduce3OpFactory<T> *reduce3OpFactory;

	if(threadIdx.x == 0)
		reduce3OpFactory = new functions::reduce3::Reduce3OpFactory<T>();
	__syncthreads();

	if(threadIdx.x == 0)
		op = reduce3OpFactory->getOp(opNum);
	__syncthreads();
	op->transform(n,dx,xShapeInfo,dy,yShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);
	if(threadIdx.x == 0) {
		free(op);
		free(reduce3OpFactory);
	}

}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C" __global__ void reduce3Double(
		int opNum,
		int n, double *dx, int *xShapeInfo,
		double *dy,
		int *yShapeInfo, double *extraParams, double *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	reduce3Generic<double>(opNum,n,dx,xShapeInfo,dy,yShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);

}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C" __global__ void reduce3Float(
		int opNum,
		int n, float *dx, int *xShapeInfo,
		float *dy,
		int *yShapeInfo, float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	reduce3Generic<float>(opNum,n,dx,xShapeInfo,dy,yShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);

}

#endif



#endif /* REDUCE3_H_ */
