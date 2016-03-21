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
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace functions {
    namespace pairwise_transforms {
#define MIN 1e-12

/**
 * Transforms involving 2 arrays
 */
        template<typename T>
        class PairWiseTransform : public virtual functions::ops::Op<T> {
        protected:
            bool requiresSpecial = false;
        public:
            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)

#endif
            T op(T d1, T d2, T *params) = 0;

            virtual
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)

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
                    int *indexes,
                    int *yIndexes,
                    int *resultIndexes) {
                int totalThreads = gridDim.x * blockDim.x;
                int tid = threadIdx.x;
                int i = blockIdx.x * blockDim.x + tid;
                int n = shape::length(xShapeBuffer);
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
                    int *indexes,
                    int *yIndexes) {
                transform(dx,
                          xShapeBuffer,
                          y,
                          yShapeBuffer,
                          result,
                          resultShapeBuffer,
                          extraParams,
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
                    T *extraParams) {
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

                int xElementWiseStride = shape::elementWiseStride(xShapeBuffer); //shape::computeElementWiseStride(xRank,xShape,xStride,xOrder == 'f');
                int yElementWiseStride = shape::elementWiseStride(yShapeBuffer); //shape::computeElementWiseStride(yRank,yShape,yStride,yOrder == 'f');
                int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer); // shape::computeElementWiseStride(resultRank,resultShape,resultStride,resultOrder == 'f');

//                if (threadIdx.x == 0 && blockIdx.x == 0)
//                    printf("xEWStride: [%i], yEWStride: [%i], zEWStride: [%i], xLength: [%i], yLength: [%i], xOrder: [%c], yOrder: [%c]\n", xElementWiseStride, yElementWiseStride, resultElementWiseStride,shape::length(xShapeBuffer), shape::length(yShapeBuffer), xOrder, yOrder);


                int n = shape::length(xShapeBuffer);
                if(xElementWiseStride >= 1 && yElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeBuffer) == shape::order(yShapeBuffer) && shape::order(resultShapeBuffer) == shape::order(xShapeBuffer)) {
                    transform(
                            n,
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

                        int xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
                        int yOffset2 = shape::getOffset(0, yShape, yStride, yIdx, yRank);
                        int resultOffset2 = shape::getOffset(0, resultShape, resultStride, resultIdx, resultRank);
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
                    T *dx,
                    T *dy,
                    int incx,
                    int incy,
                    T *params,
                    T *result,
                    int incz) {
                int totalThreads = gridDim.x * blockDim.x;
                int tid = threadIdx.x;
                int i = blockIdx.x * blockDim.x + tid;

                if (incy == 0) {
                    if ((blockIdx.x == 0) && (tid == 0)) {
#pragma unroll
                        for (; i < n; i++) {
                            result[i * incz] = op(dx[i * incx], params);
                        }

                    }
                } else if ((incx == incy) && (incx > 0)) {
                    /* equal, positive, increments */
                    if (incx == 1) {
                        /* both increments equal to 1 */
#pragma unroll
                        for (; i < n; i += totalThreads) {
                            result[i * incz] = op(dx[i * incx], dy[i * incy],
                                                  params);
                        }
                    } else {
                        /* equal, positive, non-unit increments. */
#pragma unroll
                        for (; i < n; i += totalThreads) {
                            result[i * incz] = op(dx[i * incx], dy[i * incy],
                                                  params);
                        }
                    }
                } else {
                    /* unequal or nonpositive increments */
#pragma unroll
                    for (; i < n; i += totalThreads) {
                        result[i * incz] = op(dx[i * incx], dy[i * incy],
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
                    T *extraParams,
                    int *indexes,
                    int *yIndexes) {
                exec(dx,
                     xShapeBuffer,
                     y,
                     yShapeBuffer,
                     result,
                     resultShapeBuffer,
                     extraParams,
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
                    int *indexes,
                    int *yIndexes,
                    int *resultIndexes) {
                int n = shape::length(xShapeBuffer);
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
                    int *indexes) {
                int n = shape::length(xShapeBuffer);
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
            virtual void execSpecial(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams) = 0;
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
                    T *extraParams) {

                int n = shape::length(xShapeBuffer);
                int xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
                int yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);

                bool sameShape = shape::shapeEquals(shape::rank(xShapeBuffer), shape::shapeOf(xShapeBuffer),
                                                    shape::rank(yShapeBuffer), shape::shapeOf(yShapeBuffer));
                //ignore everything else
                if (this->requiresSpecial) {
                    this->execSpecial(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
                    return;
                }


                if (xElementWiseStride >= 1 &&
                    yElementWiseStride >= 1 &&
                    resultElementWiseStride >= 1 &&
                    shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
                    shape::order(resultShapeBuffer) == shape::order(xShapeBuffer) &&
                    sameShape) {
                    exec(dx,
                         xElementWiseStride,
                         y,
                         yElementWiseStride,
                         result,
                         resultElementWiseStride,
                         extraParams,
                         n);
                }
                    //not same shape
                else if (!sameShape && shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
                         shape::order(resultShapeBuffer) == shape::order(xShapeBuffer)) {
                    exec(dx,
                         xElementWiseStride,
                         y,
                         yElementWiseStride,
                         result,
                         resultElementWiseStride,
                         extraParams,
                         shape::length(yShapeBuffer));
                }

                else if (sameShape) {
                    int rank = shape::rank(xShapeBuffer);
                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *yShape = shape::shapeOf(yShapeBuffer);
                    int *resultShape = shape::shapeOf(resultShapeBuffer);

                    int *xStride = shape::stride(xShapeBuffer);
                    int *yStride = shape::stride(yShapeBuffer);
                    int *resultStride = shape::stride(resultShapeBuffer);

                    int xRank = shape::rank(xShapeBuffer);
                    int yRank = shape::rank(yShapeBuffer);
                    int resultRank = shape::rank(resultShapeBuffer);


                    char xOrder = shape::order(xShapeBuffer);
                    char yOrder = shape::order(yShapeBuffer);
                    char resultOrder = shape::order(resultShapeBuffer);

                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int yStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    if (PrepareThreeRawArrayIter<T>(rank,
                                                    xShape,
                                                    dx,
                                                    xStride,
                                                    y,
                                                    yStride,
                                                    result,
                                                    resultStride,
                                                    &rank,
                                                    shapeIter,
                                                    &dx,
                                                    xStridesIter,
                                                    &y,
                                                    yStridesIter,
                                                    &result,
                                                    resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter)
                        {
                            /* Process the innermost dimension */
                            T *xIter = dx;
                            T *yIter = y;
                            T *resultIter = result;
                            resultIter[0] = op(xIter[0], yIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_THREE_NEXT(dim,
                                                 rank,
                                                 coord,
                                                 shapeIter,
                                                 dx,
                                                 xStridesIter,
                                                 y,
                                                 yStridesIter,
                                                 result,
                                                 resultStridesIter);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                }

                else {
                    printf("Unable to execute pair wise transform\n");
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


#endif
            virtual ~PairWiseTransform() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    //no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 + d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Add() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Copy() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 / d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Divide() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                Divide() {
                }
            };



            /**
 *Set x to y
 */
            template<typename T>
            class Set: public virtual PairWiseTransform<T> {
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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Set() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                Set() {
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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    T diff = d1 - d2;
                    T absDiff = abs(diff);
                    if (absDiff > MIN)
                        return 1;
                    return 0;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Epsilon() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 == d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~EqualTo() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 != d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~NotEqualTo() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 >= d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~GreaterThanOrEqual() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 > d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~GreaterThan() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 < d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~LessThan() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 <= d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~LessThanOrEqual() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 * d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Multiply() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d2 / d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~ReverseDivide() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d2 - d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~ReverseSubtraction() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return d1 - d2;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Subtract() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return nd4j::math::nd4j_max<T>(d1,d2);
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Max() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *y,
                        int *yShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T d2, T *params) {
                    return nd4j::math::nd4j_min(d1,d2);
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~Min() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


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
                if(op == 16)
                    return new pairwise_transforms::ops::Set<T>();


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

    op->transform(dx,xShapeInfo,dy,yShapeInfo,result,resultShapeInfo,params);
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
        double *dx,
        double *dy,
        double *params,
        double *result,
        int *xShapeInfo,
        int *yShapeInfo,
        int *resultShapeInfo) {
    pairWiseTransformGeneric<double>(
            opNum,
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
        float *dx,
        float *dy,
        float *params,
        float *result,
        int *xShapeInfo,
        int *yShapeInfo,
        int *resultShapeInfo) {
    pairWiseTransformGeneric<float>(
            opNum,
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

    op->transform(
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            result,
            resultShapeInfo,
            params,
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
__global__ void pairWiseTransformDoubleIndex(
        int opNum,
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
__global__ void pairWiseTransformFloatIndex(
        int opNum,
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
    template<typename T>
    __device__ void pairWiseTransformStridedGeneric(
            int opNum,
            int n,
            T *dx,
            T *dy,
            int incx,
            int incy,
            T *params,
            T *result,
            int incz) {
        __shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
        __shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
        if (threadIdx.x == 0)
            newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
        __syncthreads();
        if (threadIdx.x == 0)
            op = newOpFactory->getOp(opNum);
        __syncthreads();
        op->transform(n, dx, dy, incx, incy, params, result, incz);

        if (threadIdx.x == 0) {
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
    __global__ void pairWiseTransformStridedDouble(
            int opNum,
            int n,
            double *dx,
            double *dy,
            int incx,
            int incy,
            double *params,
            double *result,
            int incz) {
        pairWiseTransformStridedGeneric<double>(
                opNum,
                n,
                dx,
                dy,
                incx,
                incy,
                params,
                result,
                incz);
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
    __global__ void pairWiseTransformStridedFloat(
            int opNum,
            int n,
            float *dx,
            float *dy,
            int incx,
            int incy,
            float *params,
            float *result,
            int incz) {
        pairWiseTransformStridedGeneric<float>(
                opNum,
                n,
                dx,
                dy,
                incx,
                incy,
                params,
                result,
                incz);
    }



#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
