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


        static _CUDA_G void summaryStatsReduceDouble(int op, double *dx, Nd4jLong *xShapeInfo, int xRank, double *extraParams, double *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, bool biasCorrected, int *allocationBuffer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
        static _CUDA_G void summaryStatsReduceFloat(int op, float *dx, Nd4jLong *xShapeInfo, int xRank, float *extraParams, float *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
        static _CUDA_G void summaryStatsReduceHalf(int op, float16 *dx, Nd4jLong *xShapeInfo, int xRank, float16 *extraParams, float16 *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);

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

            _CUDA_HD SummaryStatsData() {
                initialize();
            }

            // initialize to the identity element

            _CUDA_HD void initialize() {
                n = mean = M2 = M3 = M4 = bias = 0;
            }

            _CUDA_HD void initWithValue(T val) {
                n = 1;
                min = val;
                max = val;
                mean = val;
                M2 = 0;
                M3 = 0;
                M4 = 0;
                bias = 0;
            }

            _CUDA_HD void setValues(SummaryStatsData<T> *target) {
                n = target->n;
                min = target->min;
                max = target->max;
                mean = target->mean;
                M2 = target->M2;
                M3 = target->M3;
                M4 = target->M4;
                bias = target->bias;
            }

            _CUDA_HD T variance() {
                if (n <= 1)
                    return 0.0;
                return M2 / (n);
            }

            _CUDA_HD T varianceBiasCorrected() {
                if (this->n <= 1) {
                    return 0.0;
                }

                // return (M2 - nd4j::math::nd4j_pow<T>(skewness(), 2.0) / n) / (n - 1.0);
                return M2 / (n - 1.0);
            }


            _CUDA_HD T variance_n() {
                if (n <= 1)
                    return 0.0;
                return M2 / n;
            }

            _CUDA_HD T skewness() { return M2 > 0 ? nd4j::math::nd4j_sqrt<int>(n) * M3 / nd4j::math::nd4j_pow(M2, (T) 1.5) : (T) 0.0f; }

            _CUDA_HD T kurtosis() { return M2 > 0 ? n * M4 / (M2 * M2) : 0; }

            _CUDA_HD T getM2() {
                return M2;
            }

            _CUDA_HD void setM2(T m2) {
                M2 = m2;
            }

            _CUDA_HD T getM3() {
                return M3;
            }

            _CUDA_HD void setM3(T m3) {
                M3 = m3;
            }

            _CUDA_HD T getM4() {
                return M4;
            }

            _CUDA_HD void setM4(T m4) {
                M4 = m4;
            }

            _CUDA_HD T getMax() {
                return max;
            }

            _CUDA_HD void setMax(T max) {
                this->max = max;
            }

            _CUDA_HD T getMean() {
                return mean;
            }

            _CUDA_HD void setMean(T mean) {
                this->mean = mean;
            }

            _CUDA_HD T getMin() {
                return min;
            }

            _CUDA_HD void setMin(T min) {
                this->min = min;
            }

            _CUDA_HD T getN() {
                return n;
            }

            _CUDA_HD void setN(T n) {
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
            _CUDA_HD static SummaryStatsData<T> update(SummaryStatsData<T> x, SummaryStatsData<T> y,
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

            static inline _CUDA_D T startingValue(T *input) {
                return 0;
            }

            template<typename OpType>
            static _CUDA_D void aggregatePartials(SummaryStatsData<T> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements, T *extraParams);


            template<typename OpType>
	        static _CUDA_D void transform(T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);

            static _CUDA_D void transform(const int opNum, T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);


            static _CUDA_D void summaryStatsReduceGeneric(const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
            //static _CUDA_G void summaryStatsReduceT(int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);

            static _CUDA_H T execSummaryStatsReduceScalar(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams, bool biasCorrected);
            static _CUDA_H void execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo,bool biasCorrected);
            static _CUDA_H void execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, bool biasCorrected);
#endif

            static T execScalar(const int opNum, const bool biasCorrected, T *x, Nd4jLong *xShapeInfo, T *extraParams);

            static void exec(const int opNum, const bool biasCorrected, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength);

            template<typename OpType>
            static T execScalar(const bool biasCorrected, T *x, Nd4jLong *xShapeInfo, T *extraParams);


            template<typename OpType>
            static void exec(const bool biasCorrected, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength);

        };
    }
}


#endif /* SUMMARYSTATSREDUCE_H_ */