/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_
#ifdef __JNI__
#include <jni.h>
#endif
#include <op.h>
#include <omp.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <shape.h>
namespace functions {
    namespace pairwise_transforms {
#define MIN 1e-12

/**
 * Transforms involving 2 arrays
 */
        template<typename T>
        class PairWiseTransform : public virtual functions::ops::Op<T> {
        public:
            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) = 0;

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) = 0;

#ifdef __CUDACC__
            /**
     *
     */
    virtual __inline__ __device__ void transform(
            T *dx,
            int *xShapeBuffer,
            T *y,
            int *yShapeBuffer,
            T *result,
            int *resultShapeBuffer,
            T *extraParams,
            int n,
            int *indexes) {
        transform(dx,
                xShapeBuffer,
                y,
                yShapeBuffer,
                result,
                resultShapeBuffer,
                extraParams,
                n,
                indexes,
                indexes,
                indexes);
    }

    /**
     *
     */
    virtual __inline__ __device__ void transform(
            T *dx,
            int *xShapeBuffer,
            T *y,
            int *yShapeBuffer,
            T *result,
            int *resultShapeBuffer,
            T *extraParams,
            int n,
            int *indexes,
            int *yIndexes,
            int *resultIndexes) {

        int totalThreads = gridDim.x * blockDim.x;
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + tid;
        for (; i < n; i += totalThreads) {
            result[resultIndexes[i]] = op(dx[indexes[i]],y[yIndexes[i]], extraParams);
        }
    }


    /**
     *
     */
    virtual __inline__ __device__ void transform(
            T *dx,
            int *xShapeBuffer,
            T *y,
            int *yShapeBuffer,
            T *result,
            int *resultShapeBuffer,
            T *extraParams,
            int n,
            int *indexes,
            int *yIndexes) {
        transform(dx,
                xShapeBuffer,
                y,
                yShapeBuffer,
                result,
                resultShapeBuffer,
                extraParams,
                n,
                indexes,
                yIndexes,
                indexes);
    }

    /**
     *
     */
    virtual __inline__ __device__ void transform(
            T *dx,
            int *xShapeBuffer,
            T *y,
            int *yShapeBuffer,
            T *result,
            int *resultShapeBuffer,
            T *extraParams,
            int n) {

        int totalThreads = gridDim.x * blockDim.x;
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + tid;


        int *xShape = shape::shapeOf(xShapeBuffer);
        int *yShape = shape::shapeOf(yShapeBuffer);
        int *resultShape = shape::shapeOf(resultShapeBuffer);

        int *xStride = shape::stride(xShapeBuffer);
        int *yStride = shape::stride(yShapeBuffer);
        int *resultStride = shape::stride(resultShapeBuffer);

        int xRank = shape::rank(xShapeBuffer);
        int yRank = shape::rank(yShapeBuffer);
        int resultRank = shape::rank(resultShapeBuffer);

        int xOffset = shape::offset(xShapeBuffer);
        int yOffset = shape::offset(yShapeBuffer);
        int resultOffset = shape::offset(resultShapeBuffer);


        char xOrder = shape::order(xShapeBuffer);
        char yOrder = shape::order(yShapeBuffer);
        char resultOrder = shape::order(resultShapeBuffer);

        int xElementWiseStride = shape::computeElementWiseStride(xRank,xShape,xStride,xOrder == 'f');
        int yElementWiseStride = shape::computeElementWiseStride(yRank,yShape,resultShape,resultOrder == 'f');
        int resultElementWiseStride = shape::computeElementWiseStride(resultRank,resultShape,resultStride,resultOrder == 'f');

        if(xElementWiseStride >= 1 && yElementWiseStride >= 1 && resultElementWiseStride >= 1) {
            transform(
                    n,
                    xOffset,
                    yOffset,
                    resultOffset,
                    dx,
                    y,
                    xElementWiseStride,
                    yElementWiseStride,
                    extraParams,
                    result,
                    resultElementWiseStride);
        }

        else {
            for (; i < n; i += totalThreads) {
                int *xIdx = shape::ind2sub(xRank, xShape, i);
                int *yIdx = shape::ind2sub(yRank, yShape, i);
                int *resultIdx = shape::ind2sub(resultRank, resultShape, i);

                int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                int yOffset2 = shape::getOffset(yOffset, yShape, yStride, yIdx, yRank);
                int resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);
                result[resultOffset2] = op(dx[xOffset2],y[yOffset2], extraParams);

                free(xIdx);
                free(yIdx);
                free(resultIdx);
            }
        }




    }

    /**
     *
     * @param n
     * @param xOffset
     * @param yOffset
     * @param resultOffset
     * @param dx
     * @param dy
     * @param incx
     * @param incy
     * @param params
     * @param result
     * @param incz
     * @param blockSize
     */
    virtual __inline__ __device__ void transform(
            int n,
            int xOffset,
            int yOffset,
            int resultOffset,
            T *dx,
            T *dy,
            int incx,
            int incy,
            T *params,
            T *result, int incz) {

        int totalThreads = gridDim.x * blockDim.x;
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + tid;

        if (incy == 0) {
            if ((blockIdx.x == 0) && (tid == 0)) {
#pragma unroll
                for (; i < n; i++) {
                    result[resultOffset + i * incz] = op(dx[xOffset + i * incx], params);
                }

            }
        } else if ((incx == incy) && (incx > 0)) {
            /* equal, positive, increments */
            if (incx == 1) {
                /* both increments equal to 1 */
#pragma unroll
                for (; i < n; i += totalThreads) {
                    result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
                            params);
                }
            } else {
                /* equal, positive, non-unit increments. */
#pragma unroll
                for (; i < n; i += totalThreads) {
                    result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
                            params);
                }
            }
        } else {
            /* unequal or nonpositive increments */
#pragma unroll
            for (; i < n; i += totalThreads) {
                result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
                        params);
            }
        }
    }

#endif
        public:


            /**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
            virtual void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams, int n,
                    int *indexes,
                    int *yIndexes) {
                exec(dx,
                     xShapeBuffer,
                     y,
                     yShapeBuffer,
                     result,
                     resultShapeBuffer,
                     extraParams,
                     n,
                     indexes,
                     yIndexes,
                     indexes);
            }


            /**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
            virtual void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams,
                    const int n,
                    int *indexes,
                    int *yIndexes,
                    int *resultIndexes) {
#pragma omp parallel for
                for (int i = 0; i < n; i++) {
                    result[resultIndexes[i]] = op(dx[indexes[i]], y[yIndexes[i]], extraParams);

                }
            }





            /**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
            virtual void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams,
                    const int n,
                    int *indexes) {

#pragma omp parallel for
                    for (int i = 0; i < n; i++) {
                        result[indexes[i]] = op(dx[indexes[i]],y[indexes[i]], extraParams);

                    }

            }

            /**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
            virtual void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams, const int n) {


                int xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
                int yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);
                if(xElementWiseStride == 1 && yElementWiseStride == 1 && resultElementWiseStride == 1) {
                    exec(dx,
                         xElementWiseStride,
                         y,
                         yElementWiseStride,
                         result,
                         resultElementWiseStride,
                         extraParams,
                         n);
                }


                else {
                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *yShape = shape::shapeOf(yShapeBuffer);
                    int *resultShape = shape::shapeOf(resultShapeBuffer);

                    int *xStride = shape::stride(xShapeBuffer);
                    int *yStride = shape::stride(yShapeBuffer);
                    int *resultStride = shape::stride(resultShapeBuffer);

                    int xRank = shape::rank(xShapeBuffer);
                    int yRank = shape::rank(yShapeBuffer);
                    int resultRank = shape::rank(resultShapeBuffer);

                    int xOffset = shape::offset(xShapeBuffer);
                    int yOffset = shape::offset(yShapeBuffer);
                    int resultOffset = shape::offset(resultShapeBuffer);

                    char xOrder = shape::order(xShapeBuffer);
                    char yOrder = shape::order(yShapeBuffer);
                    char resultOrder = shape::order(resultShapeBuffer);


#pragma omp parallel for
                        for (int i = 0; i < n; i++) {
                            int *xIdx = shape::ind2subC(xRank, xShape, i);
                            int *yIdx = shape::ind2subC(yRank, yShape, i);
                            int *resultIdx = shape::ind2subC(resultRank, resultShape, i);

                            int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                            int yOffset2 = shape::getOffset(yOffset, yShape, yStride, yIdx, yRank);
                            int resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);
                            result[resultOffset2] = op(dx[xOffset2],y[yOffset2], extraParams);

                            free(xIdx);
                            free(yIdx);
                            free(resultIdx);

                        }

                    }


            }


            /**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
            virtual void exec(T *dx, int xStride, T *y, int yStride, T *result,
                              int resultStride, T *extraParams, const int n) {
                if (xStride == 1 && yStride == 1 && resultStride == 1) {
#pragma omp parallel for
                    for (int i = 0; i < n; i++) {
                        result[i] = op(dx[i], y[i], extraParams);
                    }


                }

                else {
#pragma omp parallel for
                    for (int i = 0; i < n; i++) {
                        result[i * resultStride] = op(dx[i * xStride],
                                                      y[i * yStride], extraParams);
                    }
                }

        }

        virtual inline
#ifdef __CUDACC__
        __host__ __device__
#endif
        void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
            //no extra params aggregation needs to happen
        }
#ifdef __CUDACC__
        inline __host__ __device__
#elif defined(__GNUC__)
        __always_inline

#endif
        virtual ~PairWiseTransform() {
        }
#ifdef __CUDACC__
        inline __host__ __device__
#elif defined(__GNUC__)
        __always_inline

#endif
        PairWiseTransform() {
        }

    };

    namespace ops {
/**
 * x + y
 */
        template<typename T>
        class Add: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("add_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 + d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Add() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Add() {
            }
        };

/**
 * Copy y to x
 */
        template<typename T>
        class Copy: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("copy_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Copy() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Copy() {
            }
        };

/**
 * Divide x / y
 */
        template<typename T>
        class Divide: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            __host__

#endif
            std::string name() {

                return std::string("div_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 / d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Divide() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Divide() {
            }
        };

/**
 * Whether 2 elements in an array
 * are epsilion equal
 */
        template<typename T>
        class Epsilon: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("eps_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                T diff = d1 - d2;
                T absDiff = abs(diff);
                if (absDiff < MIN)
                    return 1;
                return 0;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Epsilon() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Epsilon() {
            }
        };

/**
 * x == y (binary result)
 */
        template<typename T>
        class EqualTo: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("eq_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 == d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~EqualTo() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            EqualTo() {
            }
        };

/**
 * x == y (binary result)
 */
        template<typename T>
        class NotEqualTo: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("noteq_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 != d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~NotEqualTo() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            NotEqualTo() {
            }
        };



/**
 * Whether x > y
 */
        template<typename T>
        class GreaterThanOrEqual: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("gt_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 >= d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~GreaterThanOrEqual() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            GreaterThanOrEqual() {
            }
        };


/**
 * Whether x > y
 */
        template<typename T>
        class GreaterThan: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("gt_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 > d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~GreaterThan() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            GreaterThan() {
            }
        };

/**
 * Whether x < y
 */
        template<typename T>
        class LessThan: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("lt_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 < d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~LessThan() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            LessThan() {
            }
        };

/**
 * Whether x < y
 */
        template<typename T>
        class LessThanOrEqual: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("lteq_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 <= d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~LessThanOrEqual() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            LessThanOrEqual() {
            }
        };

/**
 * x * y
 */
        template<typename T>
        class Multiply: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("mul_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 * d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }

#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Multiply() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Multiply() {
            }
        };

/**
 * y / x
 */
        template<typename T>
        class ReverseDivide: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("rdiv_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d2 / d1;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~ReverseDivide() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            ReverseDivide() {
            }
        };

/**
 * y - x
 */
        template<typename T>
        class ReverseSubtraction: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("rsub_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d2 - d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~ReverseSubtraction() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            ReverseSubtraction() {
            }
        };

/**
 * x - y
 */
        template<typename T>
        class Subtract: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("sub_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return d1 - d2;
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Subtract() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Subtract() {
            }
        };


/**
 * x - y
 */
        template<typename T>
        class Max: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("max_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return nd4j::math::nd4j_max<T>(d1,d2);
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Max() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Max() {
            }
        };



/**
 * x - y
 */
        template<typename T>
        class Min: public virtual PairWiseTransform<T> {
        public:

            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#endif
            std::string name() {
                return std::string("min_strided");
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T d2, T *params) {
                return nd4j::math::nd4j_min(d1,d2);
            }

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline
#endif
            T op(T d1, T *params) {
                return d1;
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            virtual ~Min() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)
            __always_inline

#endif
            Min() {
            }
        };

    }

/**
 * Creates pair wise operations.
 */
    template<typename T>
    class PairWiseTransformOpFactory {
    public:



#ifdef __CUDACC__
        __host__ __device__
#endif
        PairWiseTransformOpFactory() {
        }

        /**
         * Create an operation
         * @param op the op number
         * 0: Add
         * 1: Copy
         * 2: Divie
         * 3: equal to
         * 4: greater than
         * 5: less than
         * 6: multiply
         * 7: reverse divide
         * 8 reverse subtract
         * 9: subtract
         * @return the operation based on the op number
         */
#ifdef __CUDACC__
        __inline__ __host__ __device__
#endif
        PairWiseTransform<T> *getOp(int op) {
            if (op == 0)
                return new pairwise_transforms::ops::Add<T>();
            else if (op == 1)
                return new pairwise_transforms::ops::Copy<T>();
            else if (op == 2)
                return new pairwise_transforms::ops::Divide<T>();
            else if (op == 3)
                return new pairwise_transforms::ops::EqualTo<T>();
            else if (op == 4)
                return new pairwise_transforms::ops::GreaterThan<T>();
            else if (op == 5)
                return new pairwise_transforms::ops::LessThan<T>();
            else if (op == 6)
                return new pairwise_transforms::ops::Multiply<T>();
            if (op == 7)
                return new pairwise_transforms::ops::ReverseDivide<T>();
            if (op == 8)
                return new pairwise_transforms::ops::ReverseSubtraction<T>();
            if (op == 9)
                return new pairwise_transforms::ops::Subtract<T>();
            if (op == 10)
                return new pairwise_transforms::ops::Epsilon<T>();
            if(op == 11)
                return new pairwise_transforms::ops::GreaterThanOrEqual<T>();
            if(op == 12)
                return new pairwise_transforms::ops::LessThanOrEqual<T>();
            if(op == 13)
                return new pairwise_transforms::ops::Max<T>();
            if(op == 14)
                return new pairwise_transforms::ops::Min<T>();
            if(op == 15)
                return new pairwise_transforms::ops::NotEqualTo<T>();
            return NULL;
        }



    };
}
}

#ifdef __CUDACC__

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		int n,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo) {
	__shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
	__shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0)
		op = newOpFactory->getOp(opNum);
	__syncthreads();

	op->transform(dx,xShapeInfo,dy,yShapeInfo,result,resultShapeInfo,params,n);
	if(threadIdx.x == 0) {
		free(op);
		free(newOpFactory);
	}

}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformDouble(
		int opNum,
		int n,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo) {
	pairWiseTransformGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformFloat(
		int opNum,
		int n,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo) {
	pairWiseTransformGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		int n,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes) {
	__shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
	__shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0)
		op = newOpFactory->getOp(opNum);
	__syncthreads();
	/*
	 * 	T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int n,
			int *indexes,
			int *yIndexes,
			int *resultIndexes
	 */
	op->transform(
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			result,
			resultShapeInfo,
			params,
			n,
			xIndexes,
			yIndexes,
			resultIndexes);

	if(threadIdx.x == 0) {
		free(op);
		free(newOpFactory);
	}

}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformDoubleIndex(
		int opNum,
		int n,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes) {
	pairWiseTransformGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo,
			xIndexes,
			yIndexes,
			resultIndexes);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformFloatIndex(
		int opNum,
		int n,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes) {
	pairWiseTransformGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo,
			xIndexes,
			yIndexes,
			resultIndexes);

}

#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
