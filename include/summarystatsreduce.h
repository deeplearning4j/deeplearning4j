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

#include <shape.h>
#include <op.h>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif


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
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData() {
                initialize();
            }

            // initialize to the identity element

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void initialize() {
                n = mean = M2 = M3 = M4 = bias =  0;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void initWithValue(T val) {
                n = 1;
                min = val;
                max = val;
                mean = val;
                M2 = 0;
                M3 = 0;
                M4 = 0;
                bias = 0;
            }
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setValues(SummaryStatsData<T> *target) {
                n = target->n;
                min = target->min;
                max = target->max;
                mean = target->mean;
                M2 = target->M2;
                M3 = target->M3;
                M4 = target->M4;
                bias = target->bias;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T variance()   {
                if(n <= 1)
                    return 0.0;
                return M2 / (n);
            }
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T varianceBiasCorrected() {
                if (this->n <= 1) {
                    return 0.0;
                }

                return (M2 - nd4j::math::nd4j_pow<T>(skewness(),2.0) / n ) / (n - 1.0);
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T variance_n() {
                if(n <= 1)
                    return 0.0;
                return M2 / n;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T skewness()   { return nd4j::math::nd4j_sqrt<int>(n) * M3 / nd4j::math::nd4j_pow(M2, (T) 1.5); }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T kurtosis()   { return n * M4 / (M2 * M2); }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getM2()  {
                return M2;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setM2(T m2) {
                M2 = m2;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getM3()  {
                return M3;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setM3(T m3) {
                M3 = m3;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getM4()  {
                return M4;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setM4(T m4) {
                M4 = m4;
            }

#ifdef __CUDACC__
            __inline__ __host__  __device__

#elif defined(__GNUC__)


#endif
            T getMax()  {
                return max;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setMax(T max) {
                this->max = max;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getMean()  {
                return mean;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setMean(T mean) {
                this->mean = mean;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getMin()  {
                return min;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setMin(T min) {
                this->min = min;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getN()  {
                return n;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            void setN(T n) {
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
        class SummaryStatsReduce: public  functions::ops::Op<T> {
        protected:
            bool biasCorrected = true;

        public:
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsReduce(bool biasCorrected) {
                this->biasCorrected = biasCorrected;
            }
            /**
             *
             * @param val
             * @param extraParams
             * @return
             */
            //an op for the kernel
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData<T> op(SummaryStatsData<T> val, T *extraParams) {
                return val;
            }

            /**
             *
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
            //calculate an update of the reduce operation
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData<T> update(SummaryStatsData<T> x, SummaryStatsData<T> y,
                                       T *extraParams) {
                if(x.n == 0 && y.n > 0)
                    return y;
                else if(x.n > 0 && y.n == 0)
                    return x;
                SummaryStatsData<T> result;
                T n  = x.n + y.n;
                T n2 = n  * n;
                T n3 = n2 * n;


                T delta  = y.mean - x.mean;
                T delta2 = delta  * delta;
                T delta3 = delta2 * delta;
                T delta4 = delta3 * delta;

                //Basic number of samples (n), min, and max
                result.n   = n;
                result.min = nd4j::math::nd4j_min(x.min, y.min);
                result.max = nd4j::math::nd4j_max(x.max, y.max);

                result.mean = x.mean + delta * y.n / n;

                result.M2  = x.M2 + y.M2;
                result.M2 += delta2 * x.n * y.n / n;

                result.M3  = x.M3 + y.M3;
                result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
                result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

                result.M4  = x.M4 + y.M4;
                result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
                result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
                result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

                return result;
            }

            /**
             *
             * @param f1
             * @param f2
             * @param extraParams
             * @return
             */
            //invoked when combining two kernels
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData<T> merge(SummaryStatsData<T> f1, SummaryStatsData<T> f2, T *extraParams) = 0;

            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T getValue(SummaryStatsData<T> val) = 0;

            /**
             *
             * @param reduction
             * @param n
             * @param xOffset
             * @param dx
             * @param incx
             * @param extraParams
             * @param result
             * @return
             */
            //post process result (for things like means etc)
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData<T> postProcess(SummaryStatsData<T> reduction, int n, int xOffset,
                                            T *dx, int incx, T *extraParams, T *result) = 0;

            /**
             *
             * @param d1
             * @param d2
             * @param extraParams
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsData<T> op(SummaryStatsData<T> d1, SummaryStatsData<T> d2, T *extraParams) = 0;

#ifdef __CUDACC__
            /**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(SummaryStatsData<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		SummaryStatsData<T> *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
#pragma unroll
			while ( floorPow2 & (floorPow2 - 1) ) {
				floorPow2 &= floorPow2 - 1;
			}

			if (tid >= floorPow2) {
				SummaryStatsData<T> prev = sPartials[tid - floorPow2];
				SummaryStatsData<T> curr = sPartials[tid];
				sPartials[tid - floorPow2] = update(prev,curr,extraParams);
			}
			__syncthreads();
		}

#pragma unroll
		for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numElements) {
				SummaryStatsData<T> curr = sPartials[tid];
				SummaryStatsData<T> next = sPartials[tid + activeThreads];
				sPartials[tid] = update(curr,next,extraParams);
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
	__inline__ __device__ void transform(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory<T> *manager) {


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
		T startingVal = this->startingValue(dx);
#pragma unroll
		for (int i = threadIdx.x; i < numElements; i += blockDim.x) {
			SummaryStatsData<T> val;
			val.initWithValue(startingVal);
			val.n = 0;
			sPartials[i] = val;
		}
		__syncthreads();

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
				xElementWiseStride =  xStride[dimension[0]];
			} else {
				xElementWiseStride = shape::elementWiseStride(xShapeInfo);
			}


			xLength = shape::length(xShapeInfo);


		}
		__syncthreads();
		if (!resultScalar) {

            __shared__ shape::TAD *tad;
            if (threadIdx.x == 0) {
                tad = new(manager->getTADSpace()) shape::TAD(); //(xShapeInfo,dimension,dimensionLength)
                tad->setExternalBuffers((void *) manager);
                tad->init(xShapeInfo,dimension,dimensionLength);
                tad->createTadOnlyShapeInfo();
            }
            __syncthreads();

		    if (dimensionLength > 1) {
                int rank = shape::rank(tad->tadOnlyShapeInfo);
                /*
                long allocSize = sizeof(int) * rank;
                int *xCoord = shape::cuMalloc(allocationBuffer, allocSize, manager);
                */
                int xCoord[MAX_RANK];

                for (int r = blockIdx.x; r < resultLength; r += gridDim.x) {
                    if (threadIdx.x == 0)
                        tad->createOffsetForBlock(r);
                    __syncthreads();

                    for(int i = threadIdx.x;i < tad->tadLength; i += blockDim.x) {
                        shape::ind2subC(rank,tad->tadShape, i, xCoord);
                        Nd4jIndex xOffset = shape::getOffset(tad->tadOffsetForBlock, tad->tadShape, tad->tadStride, xCoord, rank);

						SummaryStatsData <T> indexVal2;
					    indexVal2.initWithValue(dx[xOffset]);

                        sPartials[threadIdx.x] =  update(sPartials[threadIdx.x], op(indexVal2, extraParams), extraParams);
                    }
                    __syncthreads();
                    aggregatePartials(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength) ,extraParams);

					__syncthreads();
					if (threadIdx.x == 0) {
						result[r] = getValue(sPartials[threadIdx.x]);
					}

                }
		    } else {
                int xLength = shape::length(xShapeInfo);
				int tadLength = xLength / resultLength;


#pragma unroll
				for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
		    		if (threadIdx.x == 0)
                        tad->createOffsetForBlock(i);
                    __syncthreads();

					int indexX = tad->tadOffsetForBlock + xElementWiseStride * threadIdx.x;

					if (threadIdx.x < tad->tadLength) {
					    SummaryStatsData <T> indexVal;
				        indexVal.initWithValue(dx[indexX]);
					    sPartials[threadIdx.x] = op(indexVal, extraParams);
					}
#pragma unroll
					for (int x = threadIdx.x + blockDim.x; x < tadLength; x+= blockDim.x) {
						indexX = tad->tadOffsetForBlock + x * xElementWiseStride;
						SummaryStatsData <T> indexVal2;
					    indexVal2.initWithValue(dx[indexX]);
						sPartials[threadIdx.x] =  update(sPartials[threadIdx.x], op(indexVal2, extraParams), extraParams);
					}

					__syncthreads();
					aggregatePartials(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength) ,extraParams);

					__syncthreads();
					if (threadIdx.x == 0) {
						result[i] = getValue(sPartials[threadIdx.x]); //postProcess(sPartials[0],tadLength ,extraParams);
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

			if(xElementWiseStride >= 1) {
				for(int i = tid;i < n; i += (blockDim.x * gridDim.x)) {
                    SummaryStatsData <T> indexVal2;
				    indexVal2.initWithValue(dx[i * xElementWiseStride]);
					reduction =  update(reduction,indexVal2, extraParams);
    			}
            } else {
                int rank = shape::rank(xShapeInfo);
                /*
                long allocSize = sizeof(int) * rank;
    			int *ind2sub = shape::cuMalloc(allocationBuffer, allocSize, manager); //(int *) malloc(sizeof(int) * rank);
    			*/
    			int ind2sub[MAX_RANK];
#pragma unroll
	    		for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
    				shape::ind2sub(rank,shape::shapeOf(xShapeInfo),i,ind2sub);
                    int offset = shape::getOffset(0,xShapeInfo,shape::stride(xShapeInfo),ind2sub,rank);
    				SummaryStatsData <T> indexVal2;
					indexVal2.initWithValue(dx[offset]);
    				reduction =  update(reduction,indexVal2, extraParams);
			    }
            }

            __syncthreads();
            sPartials[threadIdx.x] = reduction;

            __syncthreads();
            aggregatePartials(&sPartials, threadIdx.x,blockDim.x ,extraParams);


            __syncthreads();
            if (gridDim.x > 1) {
				__shared__ bool amLast;
				unsigned int *tc = (unsigned *) reductionBuffer;
				int rank = shape::rank(xShapeInfo);
				tid = threadIdx.x;
				if (threadIdx.x == 0) {
					SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;
					pBuffer[blockIdx.x] = sPartials[0];
				}
				__syncthreads();
				__threadfence();

				if (tid==0) {
					unsigned int ticket = atomicInc(&tc[4096], gridDim.x);
				    amLast = (ticket == gridDim.x-1);
				}

				__syncthreads();

				if (amLast) {
					tc[4096] = 0;
					SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;

                    T startingVal = this->startingValue(dx);

        			SummaryStatsData<T> val;
		        	val.initWithValue(startingVal);
			        val.n = 0;
                    sPartials[threadIdx.x] =  val;

					if (threadIdx.x < gridDim.x)
						sPartials[threadIdx.x] =  pBuffer[threadIdx.x];


					__syncthreads();
					aggregatePartials(&sPartials, threadIdx.x,gridDim.x,extraParams);

					__syncthreads();
					if (tid == 0) {
						result[0] = getValue(sPartials[0]);
					}
				}
			} else {
				if (tid == 0) {
					unsigned int *tc = (unsigned *) reductionBuffer;
					tc[4096] = 0;
					result[0] = result[0] = getValue(sPartials[0]);
				}
			}
		}
	}



#endif

            /**
             * CPU interface
             * @param x the input
             * @param xShapeInfo the shape information for input
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfo the shape information
             * for result
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)

#endif
            void exec(T *x,
                      int *xShapeInfo,
                      T *extraParams,
                      T *result,
                      int *resultShapeInfo) {
                result[0] = this->execScalar(x,xShapeInfo,extraParams);
            }


            /**
             * CPU interface
             * @param x the input
             * @param xShapeInfo the shape information for input
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfo the shape information
             * for result
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)

#endif
            T execScalar(T *x,
                         int *xShapeInfo,
                         T *extraParams) {
                SummaryStatsData<T> startingIndex;
                startingIndex.initialize();
                int length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                if (xElementWiseStride == 1) {
#pragma omp parallel for shared(startingIndex)
                    for (int i = 0; i < length; i++) {
                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[i]);
#pragma omp critical
                        {
                            startingIndex = update(startingIndex, curr,
                                                   extraParams);
                        }

                    }

                    T finalVal = this->getValue(startingIndex);
                    return finalVal;
                } else {

#pragma omp parallel for shared(startingIndex)
                    for (int i = 0; i < length; i++) {
                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[i]);
#pragma omp critical
                        {
                            startingIndex = update(startingIndex, curr,
                                                   extraParams);
                        }

                    }

                    T finalVal = this->getValue(startingIndex);
                    return finalVal;
                }


            }


            /**
             * Dimension wise execution for CPU
             * @param x the input
             * @param xShapeInfo the shape information
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to execute along long
             * @param dimensionLength the length of the dimension
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)

#endif
            void exec(T *x,
                      int *xShapeInfo,
                      T *extraParams,
                      T *result,
                      int *resultShapeInfoBuffer,
                      int *dimension, int dimensionLength) {
                if (shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar(x, xShapeInfo, extraParams);
                    return;
                }


                shape::TAD tad(xShapeInfo,dimension,dimensionLength);
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                //no-op
                if(tad.dimensionLength < 1)
                    return;


                int resultLength = shape::length(resultShapeInfoBuffer);
                if(shape::elementWiseStride(tad.tadOnlyShapeInfo) > 0) {

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
#pragma omp  parallel  for
                    for (int i = 0; i < resultLength; i++) {
                        int offset = tad.tadOffsets[i];
                        int shapeIter[MAX_RANK];
                        int coord[MAX_RANK];
                        int dim;
                        int rankIter = rank;
                        int xStridesIter[MAX_RANK];
                        T *xPointer = x + offset;
                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[offset]);
                        comp = op(comp, extraParams);
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

                        result[i] = getValue(comp);
                    }
                }
                else {
                    int tadElementWiseStride = shape::elementWiseStride(tad.tadOnlyShapeInfo);
                    int tadLength = shape::length(tad.tadOnlyShapeInfo);
#pragma omp parallel for
                    for(int i = 0;  i < resultLength; i++) {
                        int baseOffset = tad.tadOffsets[i];
                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[baseOffset]);
#pragma omp simd
                        for(int j = 1; j < tadLength; j++) {
                            SummaryStatsData<T> comp2;
                            comp2.initWithValue(x[baseOffset + tadElementWiseStride * j]);
                            comp = update(comp, comp2, extraParams);
                        }

                        result[i] = getValue(comp);
                    }

                }
            }

            virtual
#ifdef __CUDACC__
            __host__ __device__
#endif
            T startingValue(T *input) {
                return 0;
            }

            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                //no extra params aggregation needs to happen
            }

#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)

#endif
            virtual ~SummaryStatsReduce() {
            }
#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)

#endif
            SummaryStatsReduce() {
            }

        };

        namespace ops {
/**
 * var(x)
 */
            template<typename T>
            class Variance: public  functions::summarystats::SummaryStatsReduce<T> {
            public:

                /**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> op(
                        functions::summarystats::SummaryStatsData<T> val, T *extraParams) override {
                    return val;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T getValue(SummaryStatsData<T> val) {
                    if (this->biasCorrected) {
                        T ret =  val.varianceBiasCorrected();
                        if(ret < 0)
                            return val.variance();
                        return ret;
                    }
                    return val.variance();
                }


                /**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
                //invoked when combining two kernels
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> merge(
                        functions::summarystats::SummaryStatsData<T> f1,
                        functions::summarystats::SummaryStatsData<T> f2, T *extraParams) override {
                    return this->update(f1,f2,extraParams);
                }

                /**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
                //post process result (for things like means etc)
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> postProcess(
                        functions::summarystats::SummaryStatsData<T> reduction, int n, int xOffset,
                        T *dx, int incx, T *extraParams, T *result) override {
                    return reduction;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,
                                       functions::summarystats::SummaryStatsData<T> d2, T *extraParams) override {
                    return d1;
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Variance() {
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                Variance() {
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                Variance(bool biasCorrected) {
                    this->biasCorrected = biasCorrected;
                }

            };
/**
 * std(x)
 */
            template<typename T>
            class StandardDeviation: public  functions::summarystats::SummaryStatsReduce<T> {
            public:


                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T getValue(SummaryStatsData<T> val) {
                    if (this->biasCorrected) {
                        T ret =  val.varianceBiasCorrected();
                        if(ret < 0)
                            return  nd4j::math::nd4j_sqrt(val.variance());
                        else
                            return  nd4j::math::nd4j_sqrt(ret);
                    }
                    return  nd4j::math::nd4j_sqrt(val.variance());
                }

                /**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> op(
                        functions::summarystats::SummaryStatsData<T> val, T *extraParams) override {
                    return val;
                }


                /**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
                //invoked when combining two kernels
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> merge(
                        functions::summarystats::SummaryStatsData<T> f1,
                        functions::summarystats::SummaryStatsData<T> f2, T *extraParams) override {
                    return this->update(f1,f2,extraParams);
                }

                /**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
                //post process result (for things like means etc)
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                functions::summarystats::SummaryStatsData<T> postProcess(
                        functions::summarystats::SummaryStatsData<T> reduction, int n, int xOffset,
                        T *dx, int incx, T *extraParams, T *result) override {
                    return reduction;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,
                                       functions::summarystats::SummaryStatsData<T> d2, T *extraParams) override {
                    return d1;
                }

#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~StandardDeviation() {
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                StandardDeviation() {
                }

#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)

#endif
                StandardDeviation(bool biasCorrected) {
                    this->biasCorrected = biasCorrected;
                }
            };
        }

        template<typename T>
        class SummaryStatsReduceOpFactory {
        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            SummaryStatsReduceOpFactory() {
            }

#ifdef __CUDACC__
            __inline__ __device__
            functions::summarystats::SummaryStatsReduce<T> * getOp(int op, bool biasCorrected, unsigned char *buffer) {
#else
            functions::summarystats::SummaryStatsReduce<T> * getOp(int op, bool biasCorrected) {
#endif
                if (op ==  0) {
#ifdef __CUDACC__
                    return new(buffer) ops::Variance<T>(biasCorrected);
#else
                    return new ops::Variance<T>(biasCorrected);
#endif
                } else if (op == 1) {
#ifdef __CUDACC__
                    return new(buffer) ops::StandardDeviation<T>(biasCorrected);
#else
                    return new ops::StandardDeviation<T>(biasCorrected);
#endif
                }
                return nullptr;
            }
#ifdef __CUDACC__
            __inline__ __device__
            functions::summarystats::SummaryStatsReduce<T> * getOp(int op, unsigned char * buffer) {
                return this->getOp(op,true, buffer);
            }
#else
            functions::summarystats::SummaryStatsReduce<T> * getOp(int op) {
                return this->getOp(op,true);
            }

#endif
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
		int op,
		T *dx,
		int *xShapeInfo, int xRank,
		T *extraParams,
		T *result,
		int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength, int postProcessOrNot,bool biasCorrected, int *allocationBuffer, T *reductionBuffer) {

	__shared__ functions::summarystats::SummaryStatsReduce<T> *indexReduce;
	__shared__ functions::summarystats::SummaryStatsReduceOpFactory<T> *newOpFactory;

	__shared__ UnifiedSharedMemory<T> *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory<T>();
	    manager->init(sizeof(UnifiedSharedMemory<T>), sizeof(functions::summarystats::SummaryStatsReduceOpFactory<T>), sizeof(functions::summarystats::SummaryStatsReduce<T>), sizeof(shape::TAD));

	    manager->setXSpace(xRank);
	    manager->setYSpace(0);
	    manager->setZSpace(zRank);
	    manager->setTADSpace(dimensionLength);
    }
    __syncthreads();

	__shared__ int *ptrSharedXShapeInfo;
    __shared__ int *ptrSharedZShapeInfo;

	if (xShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(xShapeInfo, manager->getXShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedXShapeInfo = manager->getXShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedXShapeInfo = nullptr;

    if (resultShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(resultShapeInfo, manager->getZShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedZShapeInfo = manager->getZShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedZShapeInfo = nullptr;

	if(threadIdx.x == 0) {
		newOpFactory = new(manager->getFactorySpace()) functions::summarystats::SummaryStatsReduceOpFactory<T>();
		indexReduce = newOpFactory->getOp(op,biasCorrected, manager->getFunctionSpace());
	}
	__syncthreads();

	indexReduce->transform(dx,ptrSharedXShapeInfo,extraParams,result,ptrSharedZShapeInfo,dimension,dimensionLength,postProcessOrNot, allocationBuffer, reductionBuffer, manager);
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
		bool biasCorrected, int *allocationBuffer, double *reductionBuffer) {
	summaryStatsReduceGeneric<double>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer);

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
		int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float *reductionBuffer) {
	summaryStatsReduceGeneric<float>(
			op,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer);

}



#endif






#endif /* SUMMARYSTATSREDUCE_H_ */
