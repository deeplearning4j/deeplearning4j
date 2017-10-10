/*
 * summarystatsreduce.h
 *
 *  Created on: Jan 19, 2016
 *      Author: agibsonccc
 */

#ifndef SUMMARYSTATSREDUCE_H_
#define SUMMARYSTATSREDUCE_H_
#include <templatemath.h>
#include <dll.h>

#include <helpers/shape.h>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define host_and_device inline __host__  __device__
#else
#define host_and_device inline
#endif

#ifdef __JNI__
#include <jni.h>
#endif

#include <ops/ops.h>
#include <op_boilerplate.h>

#include "legacy_ops.h"

namespace functions {
    namespace summarystats {


        // This example computes several statistical properties of a data
        // series in a single reduction.  The algorithm is described in detail here:
        // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        //
        // Thanks to Joseph Rhoads for contributing this example


        // structure used to accumulate the moments and other
        // statistical properties encountered so far.
        template <typename T>
        class SummaryStatsData {

        public:
            T n;
            T min;
            T max;
            T mean;
            T M2;
            T M3;
            T M4;
            T bias;

            host_and_device SummaryStatsData() {
                initialize();
            }

            // initialize to the identity element

            host_and_device void initialize() {
                n = mean = M2 = M3 = M4 = bias = 0;
            }

            host_and_device void initWithValue(T val) {
                n = 1;
                min = val;
                max = val;
                mean = val;
                M2 = 0;
                M3 = 0;
                M4 = 0;
                bias = 0;
            }

            host_and_device void setValues(SummaryStatsData<T> *target) {
                n = target->n;
                min = target->min;
                max = target->max;
                mean = target->mean;
                M2 = target->M2;
                M3 = target->M3;
                M4 = target->M4;
                bias = target->bias;
            }

            host_and_device T variance() {
                if (n <= 1)
                    return 0.0;
                return M2 / (n);
            }

            host_and_device T varianceBiasCorrected() {
                if (this->n <= 1) {
                    return 0.0;
                }

                return (M2 - nd4j::math::nd4j_pow<T>(skewness(), 2.0) / n) / (n - 1.0);
            }


            host_and_device T variance_n() {
                if (n <= 1)
                    return 0.0;
                return M2 / n;
            }

            host_and_device T skewness() { return M2 > 0 ? nd4j::math::nd4j_sqrt<int>(n) * M3 / nd4j::math::nd4j_pow(M2, (T) 1.5) : (T) 0.0f; }

            host_and_device T kurtosis() { return M2 > 0 ? n * M4 / (M2 * M2) : 0; }

            host_and_device T getM2() {
                return M2;
            }

            host_and_device void setM2(T m2) {
                M2 = m2;
            }

            host_and_device T getM3() {
                return M3;
            }

            host_and_device void setM3(T m3) {
                M3 = m3;
            }

            host_and_device T getM4() {
                return M4;
            }

            host_and_device void setM4(T m4) {
                M4 = m4;
            }

            host_and_device T getMax() {
                return max;
            }

            host_and_device void setMax(T max) {
                this->max = max;
            }

            host_and_device T getMean() {
                return mean;
            }

            host_and_device void setMean(T mean) {
                this->mean = mean;
            }

            host_and_device T getMin() {
                return min;
            }

            host_and_device void setMin(T min) {
                this->min = min;
            }

            host_and_device T getN() {
                return n;
            }

            host_and_device void setN(T n) {
                this->n = n;
            }
        };






#ifdef __CUDACC__
        // This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
		template<typename T>
		struct SharedSummaryStatsData {
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
		struct SharedSummaryStatsData<float> {
			__device__ SummaryStatsData<float> * getPointer() {
				extern __shared__ SummaryStatsData<float> s_int2[];
				return s_int2;
			}
		};
		// Following are the specializations for the following types.
		// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
		// One could also specialize it for user-defined types.

		template<>
		struct SharedSummaryStatsData<double> {
			__device__ SummaryStatsData<double> * getPointer() {
				extern __shared__ SummaryStatsData<double> s_int6[];
				return s_int6;
			}
		};
#endif

        /**
         * Standard deviation or variance 1 pass
         */
        template<typename T>
        class SummaryStatsReduce {
        public:
            //calculate an update of the reduce operation
            host_and_device static SummaryStatsData<T> update(SummaryStatsData<T> x, SummaryStatsData<T> y,
                                                              T *extraParams) {
                if ((int) x.n == 0 && (int) y.n > 0)
                    return y;
                else if ((int) x.n > 0 && (int) y.n == 0)
                    return x;
                SummaryStatsData<T> result;
                T n = x.n + y.n;
                T n2 = n  * n;
                T n3 = n2 * n;


                T delta = y.mean - x.mean;
                T delta2 = delta  * delta;
                T delta3 = delta2 * delta;
                T delta4 = delta3 * delta;

                //Basic number of samples (n), min, and max
                result.n = n;
                result.min = nd4j::math::nd4j_min(x.min, y.min);
                result.max = nd4j::math::nd4j_max(x.max, y.max);

                result.mean = x.mean + delta * y.n / n;

                result.M2 = x.M2 + y.M2;
                result.M2 += delta2 * x.n * y.n / n;

                result.M3 = x.M3 + y.M3;
                result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
                result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

                result.M4 = x.M4 + y.M4;
                result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
                result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
                result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

                return result;
            }



#ifdef __CUDACC__

            static inline __device__ T startingValue(T *input) {
                return 0;
            }

		/**
		 *
		 * @param sPartialsRef
		 * @param tid
		 * @param extraParams
		 */
template<typename OpType>
				static __device__ void aggregatePartials(SummaryStatsData<T> **sPartialsRef, int tid, int numElements, T *extraParams) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				SummaryStatsData<T> *sPartials = *sPartialsRef;
				int floorPow2 = blockDim.x;

				if (floorPow2 & (floorPow2 - 1)) {
#pragma unroll
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}

					if (tid >= floorPow2) {
						SummaryStatsData<T> prev = sPartials[tid - floorPow2];
						SummaryStatsData<T> curr = sPartials[tid];
						sPartials[tid - floorPow2] = update(prev, curr, extraParams);
					}
					__syncthreads();
				}

#pragma unroll
				for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads && tid + activeThreads < numElements) {
						SummaryStatsData<T> curr = sPartials[tid];
						SummaryStatsData<T> next = sPartials[tid + activeThreads];
						sPartials[tid] = update(curr, next, extraParams);
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
	static __inline__ __device__ void transform(
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
				Nd4jIndex *tadOffsets) {


				/**
				 * Gpu information for the problem
				 */
				int tid = blockIdx.x * blockDim.x + threadIdx.x;
				__shared__ volatile int resultScalar;

				__shared__ int xElementWiseStride;

				int numElements = blockDim.x;
				//shared memory space for storing intermediate results
				SummaryStatsData<T> *sPartials;
				//functions::summarystats::SharedSummaryStatsData<T> holder;

				sPartials = (SummaryStatsData<T> *) manager->getSharedReductionBuffer(); //holder.getPointer();
				T startingVal = startingValue(dx);

				SummaryStatsData<T> val;
				val.initWithValue(startingVal);
				val.n = 0;
				sPartials[threadIdx.x] = val;


				//length for the tad
				__shared__ volatile int xLength;

				__shared__ volatile int resultLength;


				SummaryStatsData <T> reduction;
				reduction.initWithValue(0.0);
				reduction.n = 0;
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

					int *xStride = shape::stride(xShapeInfo);
					char xOrder = shape::order(xShapeInfo);

					if (dimension != nullptr && (dimension[0] != MAX_DIMENSION && dimensionLength == 1)) {
						xElementWiseStride = xStride[dimension[0]];
					}
					else {
						xElementWiseStride = shape::elementWiseStride(xShapeInfo);
					}


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
							Nd4jIndex tadOffsetForBlock = tadOffsets[r];

							val.initWithValue(startingVal);
					        val.n = 0;
					        sPartials[threadIdx.x] = val;

							for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
								shape::ind2subC(tadRank, tadShape, i, xCoord);
								Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

								SummaryStatsData <T> indexVal2;
								indexVal2.initWithValue(dx[xOffset]);

								sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
							}
							__syncthreads();
							aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

							__syncthreads();
							if (threadIdx.x == 0) {
								result[r] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]);
							}

						}
					}
					else {


						for (int i = blockIdx.x; i < numTads; i += gridDim.x) {
							int tadOffsetForBlock = tadOffsets[i];

					        val.initWithValue(startingVal);
					        val.n = 0;
					        sPartials[threadIdx.x] = val;

							Nd4jIndex indexX = tadOffsetForBlock + (xElementWiseStride * threadIdx.x);

							if (threadIdx.x < tadLength) {
								SummaryStatsData <T> indexVal;
								indexVal.initWithValue(dx[indexX]);
								sPartials[threadIdx.x] = OpType::op(indexVal, extraParams);
							}
#pragma unroll
							for (int x = threadIdx.x + blockDim.x; x < tadLength; x += blockDim.x) {
								indexX = tadOffsetForBlock + x * tadEWS;
								SummaryStatsData <T> indexVal2;
								indexVal2.initWithValue(dx[indexX]);
								sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
							}

							__syncthreads();
							aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

							__syncthreads();
							if (threadIdx.x == 0) {
								result[i] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]); //postProcess(sPartials[0],tadLength ,extraParams);
							}
						}
					}
				}
				else if (resultScalar) {
					__shared__ int n;
					if (threadIdx.x == 0) {
						xElementWiseStride = shape::elementWiseStride(xShapeInfo);
						n = shape::length(xShapeInfo);
					}
					__syncthreads();

					if (xElementWiseStride >= 1) {
						for (Nd4jIndex i = tid; i < n; i += (blockDim.x * gridDim.x)) {
							SummaryStatsData <T> indexVal2;
							indexVal2.initWithValue(dx[i * xElementWiseStride]);
							reduction = update(reduction, indexVal2, extraParams);
						}
					}
					else {
						__shared__ int rank;
						__shared__ int *xShape;
						__shared__ int *xStride;
						if (threadIdx.x == 0) {
							rank = shape::rank(xShapeInfo);
							xShape = shape::shapeOf(xShapeInfo);
							xStride = shape::stride(xShapeInfo);
						}
						__syncthreads();

						int ind2sub[MAX_RANK];
#pragma unroll
						for (Nd4jIndex i = tid; i < n; i += blockDim.x * gridDim.x) {
							shape::ind2sub(rank, shape::shapeOf(xShapeInfo), i, ind2sub);
							Nd4jIndex offset = shape::getOffset(0, xShape, xStride, ind2sub, rank);
							SummaryStatsData <T> indexVal2;
							indexVal2.initWithValue(dx[offset]);
							reduction = update(reduction, indexVal2, extraParams);
						}
					}
					sPartials[threadIdx.x] = reduction;

					__syncthreads();
					aggregatePartials<OpType>(&sPartials, threadIdx.x, blockDim.x, extraParams);


					__syncthreads();
					if (gridDim.x > 1) {
						__shared__ bool amLast;
						unsigned int *tc = (unsigned int *)reductionBuffer;
						int rank = shape::rank(xShapeInfo);
						tid = threadIdx.x;
						if (threadIdx.x == 0) {
							SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;
							pBuffer[blockIdx.x] = sPartials[0];
						}
						__syncthreads();
						__threadfence();

						if (tid == 0) {
							unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
							amLast = (ticket == gridDim.x - 1);
						}

						__syncthreads();

						if (amLast) {
							tc[16384] = 0;
							SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;

							T startingVal = startingValue(dx);

							SummaryStatsData<T> val;
							val.initWithValue(startingVal);
							val.n = 0;
							sPartials[threadIdx.x] = val;

							for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
								sPartials[threadIdx.x] = update(sPartials[threadIdx.x], pBuffer[i], extraParams);
							}


							__syncthreads();
							aggregatePartials<OpType>(&sPartials, threadIdx.x, gridDim.x, extraParams);

							__syncthreads();
							if (tid == 0) {
								result[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
							}
						}
					}
					else {
						if (tid == 0) {
							unsigned int *tc = (unsigned *)reductionBuffer;
							tc[16384] = 0;
							result[0] = result[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
						}
					}
				}
			}


	static inline __device__ void transform(
		const int opNum,
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
		Nd4jIndex *tadOffsets) {
            DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets), SUMMARY_STATS_OPS);
	}
#endif


            static T execScalar(const int opNum,
                                const bool biasCorrected,
                                T *x,
                                int *xShapeInfo,
                                T *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams), SUMMARY_STATS_OPS);
            }


            static void exec(
                    const int opNum,
                    const bool biasCorrected,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *result,
                    int *resultShapeInfoBuffer,
                    int *dimension, int dimensionLength) {
                DISPATCH_BY_OPNUM(exec, PARAMS(biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength), SUMMARY_STATS_OPS);
            }

            template<typename OpType>
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)

#endif
            static T execScalar(
                    const bool biasCorrected,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams) {
                SummaryStatsData<T> startingIndex;
                startingIndex.initialize();
                Nd4jIndex length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                if (xElementWiseStride == 1) {
                    for (Nd4jIndex i = 0; i < length; i++) {
                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[i]);
                        startingIndex = update(startingIndex, curr,
                                               extraParams);
                    }

                    T finalVal = OpType::getValue(biasCorrected, startingIndex);
                    return finalVal;
                }
                else {
                    int xCoords[MAX_RANK];

                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    int xRank = shape::rank(xShapeInfo);


                    for (Nd4jIndex i = 0; i < length; i++) {
                        shape::ind2subC(xRank, xShape, i, xCoords);
                        Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoords, xRank);

                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[xOffset]);
                        startingIndex = update(startingIndex, curr, extraParams);
                    }

                    T finalVal = OpType::getValue(biasCorrected, startingIndex);
                    return finalVal;
                }


            }

            template<typename OpType>
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)

#endif
            static void exec(
                    const bool biasCorrected,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *result,
                    int *resultShapeInfoBuffer,
                    int *dimension, int dimensionLength) {
                if (shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                    return;
                }


                shape::TAD tad(xShapeInfo, dimension, dimensionLength);
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                //no-op
                if (tad.dimensionLength < 1)
                    return;

                int resultLength = shape::length(resultShapeInfoBuffer);
                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                if (resultLength == 1 || dimensionLength == shape::rank(xShapeInfo) || tad.wholeThing) {
                    result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                    return;
                }

                if (!(shape::elementWiseStride(tad.tadOnlyShapeInfo) > 0 && (tad.numTads == 1 || shape::isVector(tad.tadOnlyShapeInfo) ||
                                                                             shape::isScalar(tad.tadOnlyShapeInfo) || tad.wholeThing)) && !(dimensionLength > 1)) {

                    /**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */

                    int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                    int *xShape = shape::shapeOf(tadShapeShapeInfo);
                    int *xStride = shape::stride(tadShapeShapeInfo);
                    int rank = shape::rank(tadShapeShapeInfo);
#pragma omp parallel for schedule(guided) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        Nd4jIndex offset = tad.tadOffsets[i];
                        int shapeIter[MAX_RANK];
                        int coord[MAX_RANK];
                        int dim;
                        int rankIter = rank;
                        int xStridesIter[MAX_RANK];
                        T *xPointer = x + offset;
                        SummaryStatsData<T> comp;
                        comp.initWithValue(0.0);
                        if (PrepareOneRawArrayIter<T>(rankIter,
                                                      xShape,
                                                      xPointer,
                                                      xStride,
                                                      &rankIter,
                                                      shapeIter,
                                                      &xPointer,
                                                      xStridesIter) >= 0) {
                            ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                SummaryStatsData<T> comp2;
                                comp2.initWithValue(xPointer[0]);
                                comp = update(comp, comp2, extraParams);
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

                        result[i] = OpType::getValue(biasCorrected, comp);
                    }
                }
                else {
                    if (dimensionLength == 1) {
                        int tadElementWiseStride = shape::elementWiseStride(tad.tadOnlyShapeInfo);
                        int tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                        for (int i = 0; i < resultLength; i++) {
                            Nd4jIndex baseOffset = tad.tadOffsets[i];
                            SummaryStatsData<T> comp;
                            comp.initWithValue(x[baseOffset]);
// FIXME: reduction to be used here
                            for (int j = 1; j < tadLength; j++) {
                                SummaryStatsData<T> comp2;
                                comp2.initWithValue(x[baseOffset + (tadElementWiseStride * j)]);
                                comp = update(comp, comp2, extraParams);
                            }

                            result[i] = OpType::getValue(biasCorrected, comp);
                        }
                    } else {
                        int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                        int *tadShape = shape::shapeOf(tadShapeShapeInfo);
                        int *tadStride = shape::stride(tadShapeShapeInfo);
                        int tadRank = shape::rank(tadShapeShapeInfo);
                        int tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                        for (int r = 0; r < resultLength; r++) {
                            int xCoord[MAX_RANK];
                            Nd4jIndex tadOffsetForBlock = tad.tadOffsets[r];

                            SummaryStatsData<T> comp;
                            comp.initWithValue(x[tadOffsetForBlock]);

// FIXME: reduction should be fixed
                            for (int i = 1; i < tadLength; i ++) {
                                shape::ind2subC(tadRank, tadShape, i, xCoord);
                                Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                                SummaryStatsData <T> indexVal2;
                                indexVal2.initWithValue(x[xOffset]);

                                comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
                            }
                            result[r] = OpType::getValue(biasCorrected, comp);
                        }
                    }
                }
            }
        };
    }
}


#ifdef __CUDACC__
/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along long
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
template <typename T>
__device__ void summaryStatsReduceGeneric(
		const int op,
		T *dx,
		int *xShapeInfo, int xRank,
		T *extraParams,
		T *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength, int postProcessOrNot,bool biasCorrected, int *allocationBuffer, T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::summarystats::SummaryStatsReduce<T>), sizeof(shape::TAD), xRank);
	}
	__syncthreads();

	functions::summarystats::SummaryStatsReduce<T>::transform(
		op,
	    dx,
	    xShapeInfo,
	    extraParams,
	    result,
	    resultShapeInfo,
	    dimension,
	    dimensionLength,
	    biasCorrected,
	    allocationBuffer,
	    reductionBuffer,
	    manager,
	    tadOnlyShapeInfo,
	    tadOffsets);
}

/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along long
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
__global__ void summaryStatsReduceDouble(
		int op,
		double *dx,
		int *xShapeInfo, int xRank,
		double *extraParams,
		double *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		bool biasCorrected, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
	summaryStatsReduceGeneric<double>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along long
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
 __global__ void summaryStatsReduceFloat(
		int op,
		float *dx,
		int *xShapeInfo, int xRank,
		float *extraParams,
		float *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
	summaryStatsReduceGeneric<float>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void summaryStatsReduceHalf(
		int op,
		float16 *dx,
		int *xShapeInfo, int xRank,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
	summaryStatsReduceGeneric<float16>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}



#endif






#endif /* SUMMARYSTATSREDUCE_H_ */