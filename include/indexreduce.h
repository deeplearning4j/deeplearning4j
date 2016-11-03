/*
 * indexreduce.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include <shape.h>
#ifndef __CUDACC__
#include <omp.h>
#endif
#include <dll.h>
#include <ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif


#include <pairwise_util.h>


#define INDEX_REDUCE_OPS \
        (0, simdOps::IndexMax), \
        (1, simdOps::IndexMin)
        

namespace functions {
	namespace indexreduce {

#ifdef __CUDACC__
		// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template<typename T>
struct SharedIndexValue {
	// Ensure that we won't compile any un-specialized types
	__device__ T * getPointer() {
		extern __device__ void error(void);
		error();
		return 0;
	}
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<float> {
	__device__ IndexValue<float> * getPointer() {
		extern __shared__ IndexValue<float> s_int2[];
		return s_int2;
	}
};
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<double> {
	__device__ IndexValue<double> * getPointer() {
		extern __shared__ IndexValue<double> s_int6[];
		return s_int6;
	}
};
#endif

		template<typename T>
		class IndexReduce {

		public:
#ifdef __CUDACC__

		static inline __device__ void transform(
			const int opNum,
			T *x,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot,
			int *allocationBuffer,
			T *reductionBuffer,
			UnifiedSharedMemory *manager,
			int *tadShapeInfo,
			int *tadOffset) {
                    DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, manager, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
		}

			/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
template<typename OpType>
	static inline __device__ void aggregatePartials(IndexValue<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		IndexValue<T> *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
			while ( floorPow2 & (floorPow2 - 1) ) {
				floorPow2 &= floorPow2 - 1;
			}

			if (tid >= floorPow2) {
				IndexValue<T> prev = sPartials[tid - floorPow2];
				IndexValue<T> curr = sPartials[tid];
				sPartials[tid - floorPow2] = OpType::update(prev,curr,extraParams);
			}
			__syncthreads();
		}

#pragma unroll
		for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numElements) {
				IndexValue<T> curr = sPartials[tid];
				IndexValue<T> next = sPartials[tid + activeThreads];
				sPartials[tid] = OpType::update(curr,next,extraParams);
			}
			__syncthreads();
		}

	}

	/**
	 * @param n n is the number of
	 *        elements to loop through
	 * @param dx the data to operate on
	 * @param xVectorInfo the meta data for the vector:
	 *                              0 is the offset
	 *                              1 is the increment/stride
	 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
	 *                              3 is the element wise stride for the buffer
	 *                              4 is the number of elements it takes to get to the next row/column/tensor
	 * @param gpuInformation
	 *                              0 is the block size
	 *                              1 is the grid size
	 *                              2 is the shared memory size
	 * @param problemDefinition
	 *                          0 is the number of elements per vector
	 *                          1 is the number of vectors
	 */
template<typename OpType>
	static inline __device__ void transform(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot,
			int *allocationBuffer,
			T *reductionBuffer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo,
			int *tadOffsets) {
		/**
		 * Gpu information for the problem
		 */
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ volatile int resultScalar;

		//shared memory space for storing intermediate results
		IndexValue<T> *sPartials;


		sPartials = (IndexValue<T> *)manager->getSharedReductionBuffer(); //holder.getPointer();
		T startingVal = OpType::startingValue(dx);


		IndexValue <T> val = {startingVal, threadIdx.x};
		sPartials[threadIdx.x] = val;

		//length for the tad
		__shared__ volatile int xLength;

		__shared__ volatile int resultLength;



		//only compute the tad indexes once
		IndexValue <T> reduction = {startingVal, 0};

		if (threadIdx.x == 0) {
			if (resultShapeInfo != nullptr)
				resultLength = shape::length(resultShapeInfo);
			else resultLength = 1;

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

		//	xElementWiseStride = shape::elementWiseStride(xShapeInfo);

			xLength = shape::length(xShapeInfo);
		}
		__syncthreads();

		if (!resultScalar) {

			__shared__ int tadLength;
            __shared__ int tadEWS;
            __shared__ int tadRank;
            __shared__ int numTads;
            __shared__ int *tadShape;
            __shared__ int *tadStride;
            if (threadIdx.x == 0) {
          	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                tadRank = shape::rank(tadOnlyShapeInfo);
                numTads = shape::length(xShapeInfo) / tadLength;

                tadShape = shape::shapeOf(tadOnlyShapeInfo);
                tadStride = shape::stride(tadOnlyShapeInfo);
            }
            __syncthreads();

			if (dimensionLength > 1) {
                int xCoord[MAX_RANK];

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					int tadOffsetForBlock = tadOffsets[r];

                    for(unsigned int i = threadIdx.x;i < tadLength; i += blockDim.x) {
                        shape::ind2subC(tadRank,tadShape, i, xCoord);
                        Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
						IndexValue<T> comp {dx[xOffset], i};

                    	sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x],OpType::op(sPartials[threadIdx.x], comp,extraParams),extraParams);
                    }

                    __syncthreads();
					aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

					__syncthreads();
					if (threadIdx.x == 0) {
						result[r] = (T) sPartials[threadIdx.x].index;
					}
				}
			} else {

#pragma unroll
				for(int i = blockIdx.x; i < numTads; i+= gridDim.x) {
					int tadOffsetForBlock = tadOffsets[i];

					sPartials[threadIdx.x] = {dx[tadOffsetForBlock], 0};
#pragma unroll
					for (unsigned int x = threadIdx.x; x < tadLength; x+= blockDim.x) {
						IndexValue<T> comp {dx[tadOffsetForBlock + x * tadEWS], x};
						sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], comp, extraParams);
					}

					__syncthreads();
					aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

					__syncthreads();
					if (threadIdx.x == 0) {
						result[i] = (T)  sPartials[threadIdx.x].index; //postProcess(sPartials[0],tadLength ,extraParams);
					}
				}
			}
		}


		//reduce to 1 result
		else if (resultScalar) {
			int n = shape::length(xShapeInfo);
			int xElementWiseStride = shape::elementWiseStride(xShapeInfo);

			if(xElementWiseStride >= 1) {
				for(int i = tid;i < n; i += (blockDim.x * gridDim.x)) {
					IndexValue <T> indexVal = {dx[i * xElementWiseStride], i};
					reduction = OpType::update(reduction, indexVal, extraParams);
				}
			} else {
				int rank = shape::rank(xShapeInfo);
				int ind2sub[MAX_RANK];
#pragma unroll
				for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
					shape::ind2sub(rank,shape::shapeOf(xShapeInfo),i,ind2sub);
					int offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),ind2sub,rank);
					IndexValue <T> indexVal = {dx[offset], i};
					reduction = OpType::update(reduction, indexVal, extraParams);
				}
			}


			sPartials[threadIdx.x] = reduction;
			__syncthreads();

			aggregatePartials<OpType>(&sPartials, threadIdx.x, blockDim.x,extraParams);
			__syncthreads();

			if (gridDim.x > 1) {
				__shared__ bool amLast;
				unsigned int *tc = (unsigned int *) reductionBuffer;
				int rank = shape::rank(xShapeInfo);
				tid = threadIdx.x;
				if (threadIdx.x == 0) {
					IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;
					pBuffer[blockIdx.x] = {sPartials[0].value, sPartials[0].index};
				}
				__threadfence();
				__syncthreads();

				if (tid==0) {
					unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
				    amLast = (ticket == gridDim.x-1);
				}

				__syncthreads();

				if (amLast) {
					tc[16384] = 0;
					IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;


					sPartials[threadIdx.x] = {0, 0};

					for (Nd4jIndex i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
                        sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], pBuffer[i], extraParams);
                    }



					__syncthreads();
					aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x),extraParams);

					__syncthreads();
					if (tid == 0) {
						result[0] = (T)  sPartials[0].index;
					}
				}
			} else {
				if (tid == 0) {
					unsigned int *tc = (unsigned *) reductionBuffer;
					tc[16384] = 0;
					result[0] = (T) sPartials[0].index;
				}
			}
		}
	}


#endif
		static T execScalar(
			const int opNum,
			T *x,
			int *xShapeInfo,
			T *extraParams) {
                    RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
		}

		static void exec(const int opNum,
			T *x,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfoBuffer,
			int *dimension,
			int dimensionLength, int *tadShapeInfo, int *tadOffset) {
                    DISPATCH_BY_OPNUM(exec, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
		}


			template<typename OpType>
#ifdef __CUDACC__
			__host__
#elif defined(__GNUC__)

#endif
			static inline T execScalar(T *x,
						 int *xShapeInfo,
						 T *extraParams) {

				T startingVal = OpType::startingValue(x);
				IndexValue<T> startingIndex;
				startingIndex.value = startingVal;
				startingIndex.index = 0;
				int length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				if(xElementWiseStride < 1) {
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];

					int *xShape = shape::shapeOf(xShapeInfo);
					int *xStride = shape::stride(xShapeInfo);
					int rank = shape::rank(xShapeInfo);
					if(PrepareOneRawArrayIter<T>(rank,
												 xShape,
												 x,
												 xStride,
												 &rank,
												 shapeIter,
												 &x,
												 xStridesIter) >= 0) {

						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							/* Process the innermost dimension */
							int i = shape::getOffset(0,xShape,xStride,coord,rank);
							IndexValue<T> curr;
							curr.value = x[i];
							curr.index = i;
							startingIndex = OpType::update(startingIndex, curr,
												   extraParams);
						} ND4J_RAW_ITER_ONE_NEXT(dim,
												 rank,
												 coord,
												 shapeIter,
												 x,
												 xStridesIter);
						return startingIndex.index;
					}
					else {
						printf("Unable to prepare array\n");
					}

				}
				else {

					if (xElementWiseStride == 1) {
						if(length < ELEMENT_THRESHOLD) {
// FIXME: proper reduction to be used here
//#pragma omp simd
							for (Nd4jIndex i = 0; i < length; i++) {
								IndexValue<T> curr;
								curr.value = x[i];
								curr.index = i;
								startingIndex = OpType::update(startingIndex, curr,
													   extraParams);



							}
							return startingIndex.index;
						}
						else {
							BlockInformation info(length, ELEMENT_THRESHOLD);

#pragma omp parallel num_threads(info.threads) if (info.threads > 1) default(shared)

							{
								IndexValue<T> local;
								local.value = OpType::startingValue(x);
								local.index = 0;

								for (Nd4jIndex i = omp_get_thread_num(); i < info.chunks; i+= info.threads) {
									int newOffset = (i * info.items);
									T *chunk = x + newOffset;
									int itemsToLoop = info.items;
									if(newOffset >= length) {
										break;
									}

									//handle modulo case
									if(newOffset + info.items >= length) {
										itemsToLoop = length - newOffset;
									}

									for (Nd4jIndex j = 0; j < itemsToLoop; j++) {
										IndexValue<T> curr;
										curr.value = chunk[j];
										curr.index = j;
										local = OpType::update(local, curr, extraParams);
									}


#pragma omp critical
									{
										startingIndex = OpType::update(startingIndex, local,
															   extraParams);
									}


								}
							}

							return startingIndex.index;
						}

					}

					else {
						for (Nd4jIndex i = 0; i < length; i++) {
							IndexValue<T> curr;
							curr.value = x[i * xElementWiseStride];
							curr.index = i;
							startingIndex = OpType::update(startingIndex, curr,
												   extraParams);
						}
					}
				}

				return  startingIndex.index;
			}

			
			template<typename OpType>
#ifdef __CUDACC__
			__host__

#elif defined(__GNUC__)

#endif
			static inline void exec(T *x,
					  int *xShapeInfo,
					  T *extraParams,
					  T *result,
					  int *resultShapeInfoBuffer,
					  int *dimension,
					  int dimensionLength, int *tadShapeInfo, int *tadOffset) {

				if(shape::isScalar(resultShapeInfoBuffer)) {
					result[0] = execScalar<OpType>(x,xShapeInfo,extraParams);
					return;
				}

				const int resultLength = shape::length(resultShapeInfoBuffer);
				IndexValue<T> *startingIndex = new IndexValue<T>[resultLength];

#pragma omp parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
				for (Nd4jIndex i = 0; i < resultLength; i++) {
					IndexValue<T> val;
					val.value = OpType::startingValue(x);
					val.index = 0;
					startingIndex[i] = val;
				}

				int *tadOnlyShapeInfo = tadShapeInfo;
				int *tadOffsets = tadOffset;
				shape::TAD *tad = nullptr;

				if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
					tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
					tad->createTadOnlyShapeInfo();
					tad->createOffsets();

					if (tad->dimensionLength < 1) {
						delete tad;
						delete[] startingIndex;
						return;
					}

					tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
					tadOffsets = tad->tadOffsets;
				}

				int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
				int numTads = shape::length(xShapeInfo) / tadLength;


				if(!(shape::elementWiseStride(tadOnlyShapeInfo) > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo)))) {
					/**
                                 * The element wise stride belong longs to a reduction index.
                                 * When used out of order, we can get rid of the data
                                 * dependencies and rely on using the max dimension
                                 * specified for stride instead.
                                 * Say we take the sum(0,1) along long arr
                                 * we can use arr.stride(1) as a representation
                                 * along long which to iterate.
                                 */

					int *tadShapeShapeInfo = tadOnlyShapeInfo;
					int *xShape = shape::shapeOf(tadShapeShapeInfo);
					int *xStride = shape::stride(tadShapeShapeInfo);
					int rank = shape::rank(tadShapeShapeInfo);

#pragma omp  parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
					for(Nd4jIndex i = 0; i < resultLength; i++) {
						int offset = tadOffsets[i];
						int shapeIter[MAX_RANK];
						int coord[MAX_RANK];
						int dim;
						int rankIter = rank;
						int xStridesIter[MAX_RANK];
						T *xPointer = x + offset;
						IndexValue<T> indexValue;
						indexValue.index = 0;
						indexValue.value = x[offset];
						if(PrepareOneRawArrayIter<T>(rankIter,
													 xShape,
													 xPointer,
													 xStride,
													 &rankIter,
													 shapeIter,
													 &xPointer,
													 xStridesIter) >= 0) {
							ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
								/* Process the innermost dimension */
								IndexValue<T> comp;
								comp.index = shape::sub2Ind(rank,xShape,coord);
								comp.value = xPointer[0];
								indexValue = OpType::update(indexValue,comp,extraParams);
							} ND4J_RAW_ITER_ONE_NEXT(dim,
													 rank,
													 coord,
													 shapeIter,
													 xPointer,
													 xStridesIter);
						}
						else {
							printf("Unable to prepare array\n");
						}

						result[i] = indexValue.index;
					}
				} else {
					int tadElementWiseStride = shape::elementWiseStride(tadOnlyShapeInfo);
					const int tadLength = shape::length(tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
					for(Nd4jIndex i = 0;  i < resultLength; i++) {
						int baseOffset = tadOffsets[i];
						IndexValue<T> indexValue;
						indexValue.index = 0;
						indexValue.value = x[baseOffset];

// FIXME: proper reduction required here
						for(int j = 1; j < tadLength; j++) {
							IndexValue<T> comp;
							comp.index = j;
							comp.value = x[baseOffset + tadElementWiseStride * j];
							indexValue = OpType::update(indexValue,comp,extraParams);
						}
						result[i] = indexValue.index;
					}
				}

				delete[] startingIndex;
			}
		};
	}


}


#ifdef __CUDACC__

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__device__ void indexReduceGeneric(
		const int op,
		T *dx,
		int *xShapeInfo, int xRank,
		T *extraParams,
		T *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::indexreduce::IndexReduce<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

	functions::indexreduce::IndexReduce<T>::transform(
			op,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationBuffer,
			reductionBuffer,
			manager,
			tadOnlyShapeInfo,
			tadOffsets);
}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
__global__ void indexReduceDouble(
		int op,
		double *dx,
		int *xShapeInfo, int xRank,
		double *extraParams,
		double *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets) {
	indexReduceGeneric<double>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
__global__ void indexReduceFloat(
		int op,
		float *dx,
		int *xShapeInfo, int xRank,
		float *extraParams,
		float *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,  int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets) {
	indexReduceGeneric<float>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void indexReduceHalf(
		int op,
		float16 *dx,
		int *xShapeInfo, int xRank,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,  int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets) {
	indexReduceGeneric<float16>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}



#endif

#endif /* INDEXREDUCE_H_ */

