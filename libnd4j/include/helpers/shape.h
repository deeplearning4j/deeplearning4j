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

/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include <cstring>
#include <cstdio>
#include "../dll.h"
#include "../nd4jmalloc.h"
#include "../templatemath.h"
#include "../helpers/logger.h"
#include "../pointercast.h"
#include "../cnpy/cnpy.h"
#include <op_boilerplate.h>

#define MAX_DIMENSION 0x7fffffff
#define MAX_NUM_THREADS  1024
#define MAX_RANK 32
#define MAX_COORD 3
#define PREALLOC_SIZE 33554432
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/sharedmem.h>
#endif


#ifdef __CUDACC__
#define INLINEDEF inline
#else
#define INLINEDEF inline
#endif

#include "../pairwise_util.h"
#include <stdint.h>
#include <array/ArrayOptions.h>

namespace shape {

/**
 * Shape information approximating
 * the information on an ndarray
 */
    struct ND4J_EXPORT ShapeInformation {
        _CUDA_HD ShapeInformation(Nd4jLong *shape_ = nullptr, Nd4jLong *stride_ = nullptr, char order_ = 0, int rank_ = 0, int offset_ = 0, int elementWiseStride_ = 0)
                : shape(shape_), stride(stride_), order(order_), rank(rank_), offset(offset_), elementWiseStride(elementWiseStride_)
        {}

        Nd4jLong *shape;
        Nd4jLong *stride;
        char order;
        int rank;
        int offset;
        int elementWiseStride;
    };

/**
 * Indexing information
 * for bounds checking
 */
    struct ND4J_EXPORT CurrentIndexing {
        int numElementsPerThread;
        int blockStartingIndex;
        int startingThreadIndex;
        int endingThreadIndex;

    };



    ND4J_EXPORT _CUDA_HD bool shapeEquals(int shape1Rank,Nd4jLong *shape1,int shape2Rank,Nd4jLong *shape2);

    ND4J_EXPORT _CUDA_HD Nd4jLong* detachShape(Nd4jLong *originalShape);

    ND4J_EXPORT _CUDA_HD Nd4jLong* copyShape(Nd4jLong *originalShape);

    ND4J_EXPORT _CUDA_HD bool shapeEquals(Nd4jLong *shapeInfo1,Nd4jLong *shapeInfo2);

    ND4J_EXPORT _CUDA_HD bool strideEquals(int shape1Rank,Nd4jLong *shape1,int shape2Rank,Nd4jLong *shape2);

    ND4J_EXPORT _CUDA_HD bool strideEquals(Nd4jLong *shapeInfo1,Nd4jLong *shapeInfo2);

    ND4J_EXPORT _CUDA_HD bool strideEquals(Nd4jLong *stride1,int rank1,Nd4jLong *stride2,int rank2);

    ND4J_EXPORT _CUDA_HD bool equalsSoft(Nd4jLong *shapeA, Nd4jLong *shapeB);

    ND4J_EXPORT _CUDA_HD bool equalsStrict(Nd4jLong *shapeA, Nd4jLong *shapeB);

    ND4J_EXPORT _CUDA_HD int sizeAt(const Nd4jLong *shape, const int dim);

    template <typename T>
    ND4J_EXPORT _CUDA_HD void fill(T* buffer, T value, Nd4jLong length);

    ND4J_EXPORT _CUDA_HD void traceNew(int id);


    ND4J_EXPORT _CUDA_HD int tadIndexForLinear(int linearIndex, int tadLength);

    ND4J_EXPORT _CUDA_HD int tadLength(Nd4jLong *shapeInfo, int *dimension, int dimensionLength);

    ND4J_EXPORT _CUDA_HD bool canReshape(const int oldRank, Nd4jLong* oldShape, const int newRank, Nd4jLong* newShape, bool isFOrder);

    ND4J_EXPORT _CUDA_HD bool reshapeCF(const int oldRank, Nd4jLong* oldShape, const int newRank, Nd4jLong* newShape, bool isFOrder, Nd4jLong* target);

    /**
    * Get the shape info buffer
    * for the given rank and shape.
    */
    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBuffer(int rank, nd4j::DataType dtype, Nd4jLong *shape);

    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBuffer(int rank, nd4j::DataType dtype, Nd4jLong *shape, Nd4jLong *buffer);

    /**
    * Get the shape info buffer
    * for the given rank and shape.
     */
    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBufferFortran(int rank, nd4j::DataType dtype, Nd4jLong *shape);

    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBufferFortran(int rank, nd4j::DataType dtype, Nd4jLong *shape, Nd4jLong *output);

    //ND4J_EXPORT _CUDA_HD void doPermuteShapeBuffer(Nd4jLong *shapeBuffer, int* rearrange, Nd4jLong *tmpBuffer);

    ND4J_EXPORT _CUDA_HD void doPermuteShapeBuffer(int rank, Nd4jLong *shapeBuffer, int *rearrange, Nd4jLong *tmpBuffer);

#ifdef __CUDACC__
    template <typename T>
    __device__ ND4J_EXPORT Nd4jLong *cuMalloc(Nd4jLong *buffer, long size, UnifiedSharedMemory *manager);


    __device__ ND4J_EXPORT Nd4jLong *cuMalloc(Nd4jLong *buffer, long size);
#endif



/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank);

    ND4J_EXPORT _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank, Nd4jLong* ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */

    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank);

    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank, Nd4jLong* ret);

    ND4J_EXPORT _CUDA_HD void updateStrides(const Nd4jLong *shape, const char order);


// check whether input dimensions are permuted, not permuted dimensions order have to be 0,....,rank-1
    template <typename T>
    ND4J_EXPORT _CUDA_HD bool isDimPermuted(const T* dimensions, const int dimSize);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStridesFortran(Nd4jLong *shape, int rank, int startNum);

    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStridesFortran(Nd4jLong *shape, int rank, int startNum, Nd4jLong* ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank, int startNum);

    ND4J_EXPORT _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank, int startNum, Nd4jLong* ret);

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
    ND4J_EXPORT _CUDA_HD ShapeInformation *shapeCopy( ShapeInformation *toCopy);


    ND4J_EXPORT _CUDA_HD bool strideDescendingCAscendingF(Nd4jLong *shapeBuffer);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return -1 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
    ND4J_EXPORT _CUDA_HD int computeElementWiseStride(int rank, Nd4jLong *shape, Nd4jLong *stride, int isFOrder);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return -1 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
    ND4J_EXPORT _CUDA_HD int computeElementWiseStride(int rank, Nd4jLong *shape, Nd4jLong *stride, int isFOrder, Nd4jLong *dimension, int dimensionLength);

    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeInfoOnlyShapeAndStride(Nd4jLong *shapeInfo, Nd4jLong *dimension, int dimensionLength,bool reverseCopyStride);

    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeInfoOnlyShapeAndStride(Nd4jLong *shapeInfo, Nd4jLong *dimension, int dimensionLength,bool reverseCopyStride, Nd4jLong *buffer);
/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *doPermuteSwap(int length, Nd4jLong *shape, int* rearrange);



/**
 * In place permute swap
 * @param length
 * @param shape
 * @param rearrange
 */
    ND4J_EXPORT _CUDA_HD void doPermuteSwap(int length, Nd4jLong **shape, int* rearrange);

    ND4J_EXPORT _CUDA_HD Nd4jLong *permuteShapeBuffer(Nd4jLong *shapeBuffer, int* rearrange);

    ND4J_EXPORT _CUDA_HD void permuteShapeBufferInPlace(Nd4jLong *shapeBuffer, int* rearrange, Nd4jLong *out);

    ND4J_EXPORT _CUDA_HD void doPermuteShapeInfo(Nd4jLong *shapeBuffer, const int *rearrange);

    ND4J_EXPORT _CUDA_HD void doPermuteShapeInfo(Nd4jLong *shapeBuffer, const Nd4jLong *rearrange);

    ND4J_EXPORT _CUDA_HD void doPermuteShapeBuffer(Nd4jLong *shapeBuffer, int* rearrange);

    ND4J_EXPORT _CUDA_HD void doPermuteShapeBuffer(int rank,Nd4jLong *shapeBuffer, int* rearrange);
    /**
     * Rearrange the permute indexes
     * according to which  dimensions are specified.
     *
     * For example, dimension is implicitly:
     * 0,1,2
     *
     * If you want to do a reduce along dimensions 0 and 1,
     * you need to permute the indexes to be:
     * 2,0,1
     *
     * which will give us the ability to ierate along an element
     * wise stride.
     */

    ND4J_EXPORT _CUDA_HD Nd4jLong* createPermuteIndexes(int originalRank, int *dimension,int dimensionLength);

    ND4J_EXPORT _CUDA_HD Nd4jLong* computeResultShape(Nd4jLong *originalShapeBuffer, int *dimension,int dimensionLength);

    /**
     * This method does inplace transpose of given shapeBuffer
     *
     * @param shapeBuffer
     */
    ND4J_EXPORT _CUDA_HD void transposeInplace(Nd4jLong *shapeBuffer);


/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
    ND4J_EXPORT _CUDA_HD char getOrder(int length, Nd4jLong *shape, Nd4jLong *stride, int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
    template <typename T>
    ND4J_EXPORT _CUDA_HD int checkArrangeArray(T *arr, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
    ND4J_EXPORT _CUDA_HD void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of cthe shape
 */
    ND4J_EXPORT _CUDA_HD int isVector(Nd4jLong *shape, int rank);


    /**
     * When 1 dimension is the whole length of the
     * array
     */
    ND4J_EXPORT _CUDA_HD int oneDimEqualToLength(Nd4jLong *shape, int rank);

    ND4J_EXPORT _CUDA_HD int oneDimEqualToLength(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD int isVector(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD bool isLikeVector(Nd4jLong *shapeInfo, int& posOfNonUnityDim);

    ND4J_EXPORT _CUDA_HD bool isRowVector(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD bool isColumnVector(Nd4jLong *shapeInfo);
    /**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */

    ND4J_EXPORT _CUDA_HD int isMatrix(Nd4jLong *shape, int rank);

    INLINEDEF _CUDA_HD int isMatrix(Nd4jLong *shapeInfo);
/**
 * Returns the shape portion of an information
 * buffer
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeOf(Nd4jLong *buffer);

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */

    template <typename T>
    ND4J_EXPORT _CUDA_HD T* copyOf(Nd4jLong length, T *toCopy);

    template <typename T>
    ND4J_EXPORT _CUDA_HD T* copyOf(Nd4jLong length, T *toCopy, T *ret);

    /**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */

    template <typename T>
    ND4J_EXPORT _CUDA_HD void copyTo(Nd4jLong length, T *from, T *to);
    /**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
    ND4J_EXPORT _CUDA_HD void copyTo(int length, Nd4jLong *from, Nd4jLong *to, Nd4jLong *indexes);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
    //ND4J_EXPORT _CUDA_HD Nd4jLong *permutedStrides(Nd4jLong *toPermute, int shapeRank, Nd4jLong *rearrange);

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *slice(Nd4jLong *shape);

    ND4J_EXPORT _CUDA_HD int slices(Nd4jLong *shapeBuffer);

    ND4J_EXPORT _CUDA_HD Nd4jLong *sliceOfShapeBuffer(Nd4jLong sliceIdx, Nd4jLong *shapeBuffer);
/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
    ND4J_EXPORT _CUDA_HD int shapeInfoLength(int rank);

    ND4J_EXPORT _CUDA_HD int shapeInfoLength(Nd4jLong* shapeInfo);

    ND4J_EXPORT _CUDA_HD int shapeInfoLength(const Nd4jLong* shapeInfo);

    ND4J_EXPORT _CUDA_HD size_t shapeInfoByteLength(int rank);

    ND4J_EXPORT _CUDA_HD size_t shapeInfoByteLength(Nd4jLong* shapeInfo);

    ND4J_EXPORT _CUDA_HD size_t shapeInfoByteLength(const Nd4jLong* shapeInfo);

/**
 * Returns the rank portion of
 * an information buffer
 */
    ND4J_EXPORT _CUDA_HD int rank(const Nd4jLong *buffer);

/**
 * Converts a raw int buffer of the layout:
 * rank
 * shape
 * stride
 * offset
 * elementWiseStride
 *
 * where shape and stride are both straight int pointers
 */
    ND4J_EXPORT _CUDA_HD ShapeInformation *infoFromBuffer(Nd4jLong *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *stride(Nd4jLong *buffer);

/**
 * Compute the length of the given shape
 */
    ND4J_EXPORT _CUDA_HD bool isEmpty(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD Nd4jLong length(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD Nd4jLong length(std::initializer_list<int>& shape);

    ND4J_EXPORT _CUDA_HD Nd4jLong length(std::initializer_list<Nd4jLong>& shape);

/***
 * Returns the offset portion of an information buffer
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong offset(Nd4jLong *buffer);

    ND4J_EXPORT _CUDA_HD Nd4jLong& extra(Nd4jLong *buffer);

/**
 * Returns the ordering
 * for this shape information buffer
 */
    ND4J_EXPORT _CUDA_HD char order(const Nd4jLong *buffer);

/**
 * Returns the element wise stride for this information
 * buffer
 */
   ND4J_EXPORT _CUDA_HD Nd4jLong elementWiseStride(Nd4jLong *buffer);


    /**
 * Returns the element wise stride for this information
 * buffer
     * relative to a dimension and ordering for a reduction index
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong reductionIndexElementWiseStride(Nd4jLong *buffer, int *dimension, int dimensionLength);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
    ND4J_EXPORT _CUDA_HD int isScalar(Nd4jLong *info);

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
    ND4J_EXPORT _CUDA_HD int isScalar(volatile ShapeInformation *info);

/**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
    template <typename T1, typename T2>
    ND4J_EXPORT _CUDA_HD void removeIndex(T1 *data, T2 *indexes, Nd4jLong dataLength, Nd4jLong indexesLength, T1 *out);

    /**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */

    template <typename T1, typename T2>
    ND4J_EXPORT _CUDA_HD T1* removeIndex(T1 *data, T2 *indexes, Nd4jLong dataLength, Nd4jLong indexesLength);

    /**
     * Iterate over a given set of indexes
     * the begin and end indexes are 0 based.
     * 1 padding is automatically assumed for the ending.
     *
     * For example if you want to iterate over 0 to 4
     * it will go to 4 rather than 3.
     *
     * indexes should be the indexes to exclude
     * indexes length should be the length of indexes
     */
    ND4J_EXPORT _CUDA_HD Nd4jLong* everyIndexBut(Nd4jLong *indexes,int indexesLength,int begin,int end);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    ND4J_EXPORT int tadOffset(shape::ShapeInformation *xInfo, int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong* ensureVectorShape(Nd4jLong *shape);

    ND4J_EXPORT _CUDA_HD Nd4jLong* createScalarShapeInfo();

    ND4J_EXPORT _CUDA_HD Nd4jLong* createScalarShapeInfo(Nd4jLong *ret);

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
    template <typename T>
    ND4J_EXPORT _CUDA_HD T* range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
    template <typename T>
    ND4J_EXPORT _CUDA_HD T* range(int from, int to);

/**
 * Keep the given indexes
 * in the data
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *keep(volatile Nd4jLong *data, int* index, int indexLength, int dataLength);

/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */

    template <typename T>
    ND4J_EXPORT _CUDA_HD T* reverseCopy(T *data, Nd4jLong length);

    template <typename T>
    ND4J_EXPORT _CUDA_HD void reverseCopyTo(T *from, T *to, Nd4jLong length);

    template <typename T>
    ND4J_EXPORT _CUDA_HD void reverseCopyTo(T *from, T *to, Nd4jLong *indexes, Nd4jLong length);

    template <typename T1, typename T2>
    ND4J_EXPORT _CUDA_H void convertT(T1 *from, T2 *to, Nd4jLong length);
/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
    template <typename T>
    ND4J_EXPORT _CUDA_HD T* concat(T* arr1, Nd4jLong arr1Length, T* arr2, Nd4jLong arr2Length);

/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
    template <typename T>
    ND4J_EXPORT _CUDA_HD T* concat(int numArrays, int numTotalElements, Nd4jLong **arr, Nd4jLong *lengths);

/**
 * Get the length per slice of the
 * given shape and the dimension
 * @param rank the rank of the shape
 * @param shape the shape of to get
 * the length per slice for
 * @param dimension the dimension to
 * get the length per slice for
 * @param dimensionLength the length of the dimension array
 * @return the length per slice of the given shape
 * along the given dimension
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong lengthPerSlice(int rank, Nd4jLong *shape, int *dimension, int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong sliceOffsetForTensor(int rank,
                                       int index,
                                       Nd4jLong *shape,
                                       Nd4jLong *tensorShape,
                                       int tensorShapeLength,
                                       int *dimension,
                                       int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong sliceOffsetForTensor(int index,int tensorLength,int lengthPerSlice2);
/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
//    ND4J_EXPORT _CUDA_HD int offset(int index,
//                         int rank,
//                         shape::ShapeInformation *info,
//                         Nd4jLong *dimension,
//                         int dimensionLength);


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong tensorsAlongDimension(int rank,
                                        volatile int length,
                                        volatile Nd4jLong *shape,
                                        int *dimension,
                                        int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong tensorsAlongDimension(Nd4jLong *shapeInfo, int *dimension, int dimensionLength);



/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
    ND4J_EXPORT _CUDA_HD int tadForBlockIndex(int blockSize, int blockIdx, int i);

/**
 * Computes the number of tads per block
 *
 */
    ND4J_EXPORT _CUDA_HD int tadsPerBlock(int blockSize, int tads);

//    ND4J_EXPORT _CUDA_HD Nd4jLong *tadShapeInfo(int index, Nd4jLong *xShapeInfo, Nd4jLong *dimension,
//                                int dimensionLength);

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong *toShapeBuffer( ShapeInformation *info);

    ND4J_EXPORT _CUDA_HD Nd4jLong *toShapeBuffer( ShapeInformation *info, Nd4jLong* ret);

/**
 * Returns the number of elements per thread
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    int blockStartingIndex(int N);

/**
 * Returns the thread starting index
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    int threadStartingIndex(int N, int stride, int offset);

/**
 * Returns the thread ending index
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    int threadEndingIndex(int N, int stride, int offset);

/**
 * Returns indexing information
 * for the current kernel invocation
 */
//#ifdef __CUDACC__
//    __device__
//#endif
//    CurrentIndexing *currentIndex(int N, int offset, int stride);

/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
    ND4J_EXPORT _CUDA_HD int tadIndex(int i, int elementWiseStride, int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
    ND4J_EXPORT _CUDA_HD int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
                             int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
    ND4J_EXPORT _CUDA_HD int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
    ND4J_EXPORT _CUDA_HD int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
                                int tadNum, int originalTadNum);

/**
 * Returns the prod of the data
 * up to the given length
 */
    ND4J_EXPORT _CUDA_HD int prod(Nd4jLong *data, int length);

    ND4J_EXPORT _CUDA_HD Nd4jLong prodLong( Nd4jLong *data, int length);

    /**
     * Returns the rear most left over item not present in
     * the dimension array. This assumes that the dimension array is sorted.
     *
     * For example, given a dimension array of:
     * 0,2
     *
     * and
     *
     * 12,4,2,1 in data
     *
     * You end up with 1 (data[3])
     * since the first item won't match
     * the last item of the dimension array
     */

//    ND4J_EXPORT _CUDA_HD int rearMostLeftOverItem(Nd4jLong *data,int length,Nd4jLong *dimension,int dimensionLength);

    /**
* Get an offset for retrieval
* from a data buffer
* based on the given
* shape stride and given indices
* @param baseOffset the offset to start from
* @param shape the shape of the array
* @param stride the stride of the array
* @param indices the indices to iterate over
* @return the double at the specified index
*/
    ND4J_EXPORT _CUDA_HD Nd4jLong getOffset(Nd4jLong baseOffset,  Nd4jLong *shape,  Nd4jLong *stride,  const Nd4jLong *indices,int rank);

    ND4J_EXPORT _CUDA_HD Nd4jLong* createShapeInfo(Nd4jLong *shape, Nd4jLong *stride, int rank);

    ND4J_EXPORT _CUDA_HD Nd4jLong* createShapeInfo(Nd4jLong *shape, Nd4jLong *stride, int rank, Nd4jLong *buffer);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    ND4J_EXPORT _CUDA_HD Nd4jLong* ind2sub(int rank,  Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices);

    ND4J_EXPORT _CUDA_HD Nd4jLong *ind2sub(int rank,  Nd4jLong *shape, Nd4jLong index);

    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
    ND4J_EXPORT _CUDA_HD void  ind2sub(int rank,Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices,Nd4jLong *out);

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    ND4J_EXPORT _CUDA_HD void ind2sub(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong *out);

    /**
  * Convert a linear index to
  * the equivalent nd index
  * @param shape the shape of the dimensions
  * @param index the index to map
  * @param numIndices the number of total indices (typically prod of shape(
  * @return the mapped indexes along each dimension
  */
    ND4J_EXPORT _CUDA_HD Nd4jLong* ind2subC(int rank, Nd4jLong *shape, Nd4jLong index);
    /**
  * Convert a linear index to
  * the equivalent nd index
  * @param shape the shape of the dimensions
  * @param index the index to map
  * @param numIndices the number of total indices (typically prod of shape(
  * @return the mapped indexes along each dimension
  */
    ND4J_EXPORT _CUDA_HD Nd4jLong* ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices);

    /**
   * Convert a linear index to
   * the equivalent nd index
   * @param shape the shape of the dimensions
   * @param index the index to map
   * @param numIndices the number of total indices (typically prod of shape(
   * @return the mapped indexes along each dimension
   */
    ND4J_EXPORT _CUDA_HD void  ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices, Nd4jLong *out);

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    ND4J_EXPORT _CUDA_HD void ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong *out);

    /**
  * Convert the given index (such as 1,1)
  * to a linear index
  * @param shape the shape of the indexes to convert
  * @param indices the index to convert
  * @return the linear index given the shape
  * and indices
  */
    ND4J_EXPORT _CUDA_HD int sub2Ind(int rank, Nd4jLong *shape, Nd4jLong *indices);

    /**
   * Compute the real linear indices for the given shape and stride
   */
    ND4J_EXPORT _CUDA_HD Nd4jLong *computeIndices(int rank,  Nd4jLong *shape,  Nd4jLong *stride);

    /**
   * Compute the real linear indices for the
     * given shape buffer. Shape,stride and rank are derived
     * from the buffer
   */
    ND4J_EXPORT _CUDA_HD Nd4jLong *computeIndices( Nd4jLong *shapeBuffer);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    ND4J_EXPORT _CUDA_HD void  ind2subOrder(Nd4jLong *shapeInfo, Nd4jLong index, Nd4jLong numIndices,Nd4jLong *out);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    ND4J_EXPORT _CUDA_HD void  ind2subOrder(Nd4jLong *shapeInfo, Nd4jLong index,Nd4jLong *out);

    ND4J_EXPORT _CUDA_HD void printShapeInfo(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD void printShapeInfoLinear(Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD void printShapeInfoLinear(const char *msg, Nd4jLong *shapeInfo);

    ND4J_EXPORT _CUDA_HD void printShapeInfoLinear(const char *msg, int rank, Nd4jLong *shape, Nd4jLong *strides);

    ND4J_EXPORT _CUDA_HD void printIntArray(Nd4jLong *arr,int length);

    ND4J_EXPORT _CUDA_HD void printArray(float *arr,int length);

    template<typename T>
    ND4J_EXPORT _CUDA_HD void printArray(T *arr,int length, const char *message);

    ND4J_EXPORT _CUDA_HD Nd4jLong* shapeBufferOfNpy(int rank, unsigned int *shape,bool fortranOrder);

    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBufferOfNpy(cnpy::NpyArray arr);

//    ND4J_EXPORT _CUDA_HD Nd4jLong *shapeBufferOfNpyBuffer(char *buffer);


   // this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too big number of dimensions)
    // also sort input array of dimensions, this operation is also necessary for creating TAD object
    ND4J_EXPORT _CUDA_H void checkDimensions(const int rank, std::vector<int>& dimensions);


    // return absolute index of array min, min is sub-array of max, index to be returned is min index and corresponds to maxIdx of max array
    ND4J_EXPORT _CUDA_H Nd4jLong subArrayIndex(const Nd4jLong* maxShapeInfo, const Nd4jLong* minShapeInfo, const int maxIdx);

    ND4J_EXPORT _CUDA_HD void shapeOldScalar(nd4j::DataType dtype, Nd4jLong* const buffer, const char order);




//END HEADERS


    //BEGIN IMPLEMENTATIONS


#ifdef __CUDACC__
    template <typename T>
__device__ INLINEDEF Nd4jLong *cuMalloc(Nd4jLong *buffer, long size, UnifiedSharedMemory *manager) {
    // if we go for 3 dimensions coord space or below - just use shared memory for that
    if (size <= MAX_COORD * 4) {
        Nd4jLong *ptr = new Nd4jLong[size / 4];//manager->getSharedCoordBuffer() + (threadIdx.x * MAX_COORD);
        return ptr;
    } else {
        // otherwise go to preallocated global memory :(
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid * size > PREALLOC_SIZE - size) {
            return (Nd4jLong *) malloc(size);
        } else {
            Nd4jLong *ret = buffer;
            ret += (tid * size);
            return ret;
        }
    }
}
#endif

#ifdef __CUDACC__
    /**
* BEWARE: THIS METHOD DOES NOT CHECKS ALLOCATION BOUNDARIES
*/
__device__ INLINEDEF Nd4jLong *cuMalloc(Nd4jLong *buffer, long size) {
    Nd4jLong *ret = buffer;
    ret += (threadIdx.x * size);
    return ret;
}
#endif

/**
* Length of a tad given
* the shape information
*/
    INLINEDEF _CUDA_HD int tadLength(Nd4jLong *shapeInfo, int *dimension, int dimensionLength) {
        if(dimensionLength == 1) {
            return shape::shapeOf(shapeInfo)[dimension[0]];
        }
        else {
            int ret = 1;
            for(int i = 0; i < shape::rank(shapeInfo); i++) {
                for(int j = 0; j < dimensionLength; j++) {
                    if(i == dimension[j])
                        ret *= shape::shapeOf(shapeInfo)[dimension[j]];
                }
            }
            return ret;
        }
    }



/**
 * Tad element wise stride:
 * given the inner most dimension (the sorted dimension of the last)
 * the element wise stride of the tad (disregarding order) is the
 * last dimension's stride.
 *
 * For a given singular dimension this will just be the only entry.
 * For example, given the following c order shape/stride:
 * 2,2,3,2
 * 12,6,2,1
 *
 * The tad element wise stride for 3 will be 1.
 * For zero it wil be 12
 *
 * For 2,3 it's 1
 *
 * Note here that the multi dimensional 2,3 case
 * is equivalent to the singular 3 case.
 *
 *
 * Note that this is for the dimension that ultimately
 * ends up removed.
 *
 * Again: this may not preserve ordering of the tad
 * but maybe used for reductions.
 */
    INLINEDEF _CUDA_HD int tadElementWiseStride(Nd4jLong *shapeInfo, int *dimension,int dimensionLength) {
        return reductionIndexElementWiseStride(shapeInfo,dimension,dimensionLength);
    }


    INLINEDEF _CUDA_HD bool shapeEquals(int shape1Rank,Nd4jLong *shape1,int shape2Rank,Nd4jLong *shape2) {
        if(shape1Rank != shape2Rank)
            return false;
        //rank not equals
        for(int i = 0; i < shape1Rank; i++) {
            if(shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    INLINEDEF _CUDA_HD bool shapeEquals(Nd4jLong *shapeInfo1,Nd4jLong *shapeInfo2) {
        return shape::shapeEquals(shape::rank(shapeInfo1),shape::shapeOf(shapeInfo1),shape::rank(shapeInfo2),shape::shapeOf(shapeInfo2));
    }

    INLINEDEF _CUDA_HD bool strideEquals(int shape1Rank,Nd4jLong *shape1,int shape2Rank,Nd4jLong *shape2) {
        if(shape1Rank != shape2Rank)
            return false;
        //rank not equals
        for(int i = 0; i < shape1Rank; i++) {
            if(shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    INLINEDEF _CUDA_HD bool strideEquals(Nd4jLong *shapeInfo1,Nd4jLong *shapeInfo2) {
        return shape::strideEquals(shape::rank(shapeInfo1),shape::stride(shapeInfo1),shape::rank(shapeInfo2),shape::stride(shapeInfo2));

    }

    INLINEDEF _CUDA_HD bool strideEquals(Nd4jLong *stride1,int rank1 , Nd4jLong *stride2, int rank2) {
        if(rank1 != rank2)
            return false;

        for(int i = 0; i < rank1; i++) {
            if(stride1[i] != stride2[i])
                return false;
        }

        return true;
    }

    INLINEDEF _CUDA_HD Nd4jLong *computeResultShape(Nd4jLong *originalShapeBuffer, int* dimension,int dimensionLength) {
        Nd4jLong *retShape;
        int retShapeLength;
        if(dimensionLength == 1 && dimension[0] == 2147483647) {
            retShape = new Nd4jLong[2];
            retShape[0] = 1;
            retShape[1] = 1;
            retShapeLength = 2;
        }
        else {
            retShape = shape::removeIndex<Nd4jLong, int>(shape::shapeOf(originalShapeBuffer), dimension, shape::shapeInfoLength(shape::rank(originalShapeBuffer)), dimensionLength);
            retShapeLength =   shape::rank(originalShapeBuffer) - dimensionLength;
        }
        //ensure vector is proper shape
        if (retShapeLength == 1) {
            if (dimension[0] == 0) {
                auto newRetShape = new Nd4jLong[2]{1, retShape[0]};
                delete[] retShape;
                retShape = newRetShape;
                retShapeLength = 2;
            }
            else {
                auto newRetShape = new Nd4jLong[2]{retShape[0], 1};
                delete[] retShape;
                retShape = newRetShape;
                retShapeLength = 2;
            }
        } else if (retShapeLength == 0) {
            auto newRetShape = new Nd4jLong[2]{1, 1};
            delete[] retShape;
            retShape = newRetShape;
            retShapeLength = 2;
        }

        auto ret = shape::shapeBuffer(retShapeLength, nd4j::ArrayOptions::dataType(originalShapeBuffer), retShape);
        delete[] retShape;

        return ret;

    }

    INLINEDEF _CUDA_HD Nd4jLong *shapeInfoOnlyShapeAndStride(Nd4jLong *shapeInfo, Nd4jLong *dimension, int dimensionLength,bool reverseCopyStride, Nd4jLong *buffer) {
        Nd4jLong *theShape = shape::shapeOf(shapeInfo);
        Nd4jLong *theStride = shape::stride(shapeInfo);
        int rank = dimensionLength == 1 ? 2 : dimensionLength;
        Nd4jLong *ret = buffer;
        //set the rank
        ret[0] = rank;
        Nd4jLong *retShape = shape::shapeOf(ret);
        Nd4jLong *retStride = shape::stride(ret);
        int len = rank;

        if(dimensionLength == 1) {
            if(shape::isMatrix(theShape,shape::rank(shapeInfo))) {
                if(dimension[0] == 0) {
                    Nd4jLong newStride[2] = {theStride[dimension[0]],1};
                    Nd4jLong newShape[2] = {theShape[dimension[0]],1};
                    retShape[0] = newShape[0];
                    retShape[1] = newShape[1];
                    retStride[0] = newStride[0];
                    retStride[1] = newStride[1];
                }
                else {
                    Nd4jLong newStride[2] = {theStride[dimension[0]],1};
                    Nd4jLong newShape[2] = {theShape[dimension[0]],1};
                    retShape[0] = newShape[0];
                    retShape[1] = newShape[1];
                    retStride[0] = newStride[0];
                    retStride[1] = newStride[1];
                }
            }
            else {
                Nd4jLong newStride[2] = {1,theStride[dimension[0]]};
                Nd4jLong newShape[2] = {1,theShape[dimension[0]]};
                retShape[0] = newShape[0];
                retShape[1] = newShape[1];
                retStride[0] = newStride[0];
                retStride[1] = newStride[1];
            }



        }
        else {
            Nd4jLong *newIndexes = dimension;
            if(reverseCopyStride)
                shape::reverseCopyTo(theStride, retStride, newIndexes, len);
            else
                shape::copyTo(len, theStride, retStride, newIndexes);
            shape::copyTo(len, theShape, retShape, newIndexes);

        }


        ret[shape::shapeInfoLength(rank) - 1] = shape::order(shapeInfo);
        return ret;
    }

    INLINEDEF _CUDA_HD Nd4jLong *shapeInfoOnlyShapeAndStride(Nd4jLong *shapeInfo, Nd4jLong *dimension, int dimensionLength,bool reverseCopyStride) {
        int rank = dimensionLength == 1 ? 2 : dimensionLength;

        traceNew(4);

        Nd4jLong *ret = new Nd4jLong[shape::shapeInfoLength(rank)];
        return shapeInfoOnlyShapeAndStride(shapeInfo, dimension, dimensionLength, reverseCopyStride, ret);
    }

    INLINEDEF _CUDA_HD Nd4jLong * createShapeInfo(Nd4jLong *shape, Nd4jLong *stride, int rank) {

        traceNew(5);

        Nd4jLong *ret = new Nd4jLong[shape::shapeInfoLength(rank)];

        return createShapeInfo(shape, stride, rank, ret);
    }

    INLINEDEF _CUDA_HD Nd4jLong * createShapeInfo(Nd4jLong *shape, Nd4jLong *stride, int rank, Nd4jLong *buffer) {
        buffer[0] = rank;
        Nd4jLong *retShape = shape::shapeOf(buffer);
        Nd4jLong *retStride = shape::stride(buffer);
        for(int i = 0;i < rank; i++) {
            retShape[i] = shape[i];
            retStride[i] = stride[i];
        }

        return buffer;
    }

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    INLINEDEF _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank, int startNum) {
        if (isVector(shape, rank)) {

            traceNew(5);

            Nd4jLong *ret = new Nd4jLong[2];
            for (int i = 0; i < 2; i++)
                ret[i] = 1;
            return ret;

        }

        int dimensions = rank;

        traceNew(6);

        Nd4jLong *stride = new Nd4jLong[dimensions];
        int st = startNum;
        for (int j = 0; j < rank; j++) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }

    INLINEDEF _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank, int startNum, Nd4jLong *ret) {
        if (isVector(shape, rank)) {
            for (int i = 0; i < 2; i++)
                ret[i] = 1;
            return ret;

        }

        int dimensions = rank;

        int st = startNum;
        for (int j = 0; j < rank; j++) {
            ret[j] = st;
            st *= shape[j];
        }

        return ret;
    }

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    INLINEDEF _CUDA_HD Nd4jLong * calcStrides(Nd4jLong *shape, int rank, int startNum) {

        traceNew(7);

        Nd4jLong *stride = new Nd4jLong[rank];

        if (rank == 1) {
            stride[0] = 1;
            return stride;
        }


        if (shape::isVector(shape, rank)) {
            for (int i = 0; i < 2; i++)
                stride[i] = 1;
            return stride;

        }

        int st = startNum;
        for (int j = rank - 1; j >= 0; j--) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }

    INLINEDEF _CUDA_HD Nd4jLong * calcStrides(Nd4jLong *shape, int rank, int startNum, Nd4jLong* ret) {
        if (rank == 1) {
            ret[0] = 1;
            return ret;
        }

        if (shape::isVector(shape, rank)) {
            for (int i = 0; i < 2; i++)
                ret[i] = 1;
            return ret;

        }

        int st = startNum;
        for (int j = rank - 1; j >= 0; j--) {
            ret[j] = st;
            st *= shape[j];
        }

        return ret;
    }

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    INLINEDEF _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank) {
        return calcStridesFortran(shape, rank, 1);
    }

    INLINEDEF _CUDA_HD Nd4jLong * calcStridesFortran(Nd4jLong *shape, int rank, Nd4jLong* ret) {
        return calcStridesFortran(shape, rank, 1, ret);
    }

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    INLINEDEF _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank) {
        return calcStrides(shape, rank, 1);
    }

    INLINEDEF _CUDA_HD Nd4jLong* calcStrides(Nd4jLong *shape, int rank, Nd4jLong* ret) {
        return calcStrides(shape, rank, 1, ret);
    }

    INLINEDEF _CUDA_HD void updateStrides(Nd4jLong *shape, const char order) {
        int rank = shape[0];
        int doubleRank = 2*rank;
        
        if (rank > 0)
            if(order == 'c') {            
                shape[doubleRank] = 1;          // set unity as last stride for c order
                for(int j=1; j<rank; ++j)
                    shape[doubleRank-j] = shape[doubleRank-j+1]*shape[rank+1-j];
            }
            else {            
                shape[rank+1] = 1;             // set unity as first stride for f order
                for(int j=rank+1; j<doubleRank; ++j)
                    shape[j+1] = shape[j]*shape[j-rank];            
            }
        // set last 2 elements in shape
        shape[doubleRank + 2] = 1;
        shape[doubleRank + 3] = (int)order;
    }


// check whether input dimensions are permuted, not permuted dimensions order have to be 0,....,rank-1
    template <typename T>
    INLINEDEF _CUDA_HD bool isDimPermuted(const T* dimensions, const Nd4jLong dimSize ) {
        for(int i=0; i<dimSize-1; ++i)
            if(dimensions[i] > dimensions[i+1])
                return true;

        return false;
    }


/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
    INLINEDEF _CUDA_HD ShapeInformation *shapeCopy( ShapeInformation *toCopy) {
        auto copy = new ShapeInformation;

        traceNew(8);

        copy->shape = new Nd4jLong[toCopy->rank];

        memcpy(copy->shape, toCopy->shape, toCopy->rank * sizeof(Nd4jLong));

        traceNew(9);

        copy->stride = new Nd4jLong[toCopy->rank];
        for (int i = 0; i < toCopy->rank; i++) {
            copy->stride[i] = toCopy->stride[i];
        }
        copy->order = toCopy->order;
        copy->rank = toCopy->rank;
        copy->offset = toCopy->offset;
        copy->elementWiseStride = toCopy->elementWiseStride;
        return copy;
    }

    INLINEDEF _CUDA_HD int computeElementWiseStride(int rank, Nd4jLong *shape, Nd4jLong *stride, int isFOrder) {
        if (rank == 0)
            return 1;

        if(shape::isVector(shape,rank)) {
            return stride[rank - 1];
        }

        else {
            int oldnd;
            Nd4jLong *olddims = shape::copyOf(rank, shape);
            Nd4jLong *oldstrides = shape::copyOf(rank, stride);
            int np, op, last_stride;
            int oi, oj, ok, ni, nj, nk;

            traceNew(10);

            auto newStrides = new Nd4jLong[rank];
            oldnd = 0;
            //set the shape to be 1 x length
            int newShapeRank = 2;
            auto newShape = new Nd4jLong[newShapeRank];
            newShape[0] = 1;
            newShape[1] = shape::prodLong(shape, rank);

            /*
             * Remove axes with dimension 1 from the old array. They have no effect
             * but would need special cases since their strides do not matter.
             */
            for (oi = 0; oi < rank; oi++) {
                if (shape[oi] != 1) {
                    olddims[oldnd] = shape[oi];
                    oldstrides[oldnd] = stride[oi];
                    oldnd++;
                }
            }

            np = 1;
            for (ni = 0; ni < newShapeRank; ni++) {
                np *= newShape[ni];
            }
            op = 1;
            for (oi = 0; oi < oldnd; oi++) {
                op *= olddims[oi];
            }
            if (np != op) {
/* different total sizes; no hope */
                delete[] newStrides;
                delete[] newShape;
                delete[] oldstrides;
                delete[] olddims;
                return -1;
            }

            if (np == 0) {
/* the current code does not handle 0-sized arrays, so give up */
                delete[] newStrides;
                delete[] newShape;
                delete[] oldstrides;
                delete[] olddims;
                return -1;
            }

/* oi to oj and ni to nj give the axis ranges currently worked with */
            oi = 0;
            oj = 1;
            ni = 0;
            nj = 1;
            while (ni < newShapeRank && oi < oldnd) {
                np = newShape[ni];
                op = olddims[oi];

                while (np != op) {
                    if (np < op) {
/* Misses trailing 1s, these are handled later */
                        np *= newShape[nj++];
                    } else {
                        op *= olddims[oj++];
                    }
                }

/* Check whether the original axes can be combined */
                for (ok = oi; ok < oj - 1; ok++) {
                    if (isFOrder) {
                        if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
/* not contiguous enough */
                            delete[] newStrides;
                            delete[] newShape;
                            delete[] oldstrides;
                            delete[] olddims;
                            return -1;
                        }
                    } else {
/* C order */
                        if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
/* not contiguous enough */
                            delete[] newStrides;
                            delete[] newShape;
                            delete[] oldstrides;
                            delete[] olddims;
                            return -1;
                        }
                    }
                }

/* Calculate new strides for all axes currently worked with */
                if (isFOrder) {
                    newStrides[ni] = oldstrides[oi];
                    for (nk = ni + 1; nk < nj; nk++) {
                        newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
                    }
                } else {
/* C order */
                    newStrides[nj - 1] = oldstrides[oj - 1];
                    for (nk = nj - 1; nk > ni; nk--) {
                        newStrides[nk - 1] = newStrides[nk] * newShape[nk];
                    }
                }
                ni = nj++;
                oi = oj++;
            }

/*
 * Set strides corresponding to trailing 1s of the new shape.
 */
            if (ni >= 1) {
                last_stride = newStrides[ni - 1];
            } else {
                last_stride = stride[rank - 1];
            }
            if (isFOrder) {
                if (ni >= 1)
                    last_stride *= newShape[ni - 1];
            }
            for (nk = ni; nk < newShapeRank; nk++) {
                newStrides[nk] = last_stride;
            }
//returns the last element of the new stride array
            int ret = last_stride;
            delete[] newStrides;
            delete[] newShape;
            delete[] oldstrides;
            delete[] olddims;
            return ret;
        }


    }

    INLINEDEF _CUDA_HD int computeElementWiseStride(int rank, Nd4jLong *shape, Nd4jLong *stride, int isFOrder,
                                           Nd4jLong *dimension, int dimensionLength) {
        if(dimensionLength == 1) {
            return stride[dimension[0]];
        }
        return -1;

    }

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
    INLINEDEF _CUDA_HD Nd4jLong *shapeBuffer(int rank, nd4j::DataType dtype, Nd4jLong *shape) {
        Nd4jLong *stride = shape::calcStrides(shape, rank);

        traceNew(11);

        auto shapeInfo = new shape::ShapeInformation();
        shapeInfo->shape = shape;
        shapeInfo->stride = stride;
        shapeInfo->offset = 0;
        shapeInfo->rank = rank;
        int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);
        shapeInfo->order = 'c';
        shapeInfo->elementWiseStride = elementWiseStride;
        auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
        delete[] stride;
        delete shapeInfo;
        nd4j::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
        return shapeInfoBuffer;
    }

    /**
     * This is special method, it returns ONLY 2D shapebuffer.
     *
     * This method is used only for SoftMax
     */
    INLINEDEF _CUDA_HD Nd4jLong *shapeBuffer(int rank, nd4j::DataType dtype, Nd4jLong *shape, Nd4jLong *buffer) {
        Nd4jLong stride[MAX_RANK];
        shape::calcStrides(shape,rank, stride);


        shape::ShapeInformation shapeInfo;
        shapeInfo.shape = shape;
        shapeInfo.stride = stride;
        shapeInfo.offset = 0;
        shapeInfo.rank = rank;
        auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

        shapeInfo.order = 'c';
        shapeInfo.elementWiseStride = elementWiseStride;
        shape::toShapeBuffer(&shapeInfo, buffer);
        nd4j::ArrayOptions::setDataType(buffer, dtype);
        return buffer;
    }

/**
* Get the shape info buffer
* for the given rank and shape.
*/
    INLINEDEF _CUDA_HD Nd4jLong *shapeBufferFortran(int rank, nd4j::DataType dtype, Nd4jLong *shape) {
        auto stride = shape::calcStridesFortran(shape,rank);

        traceNew(12);

        auto shapeInfo = new shape::ShapeInformation();
        shapeInfo->shape = shape;
        shapeInfo->stride = stride;
        shapeInfo->offset = 0;
        shapeInfo->rank = rank;
        int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

        shapeInfo->order = 'f';
        shapeInfo->elementWiseStride = elementWiseStride;
        auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
        delete[] stride;
        delete shapeInfo;
        nd4j::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
        return shapeInfoBuffer;
    }

    INLINEDEF _CUDA_HD Nd4jLong *shapeBufferFortran(int rank, nd4j::DataType dtype, Nd4jLong *shape, Nd4jLong *output) {
        Nd4jLong stride[MAX_RANK];
        shape::calcStridesFortran(shape,rank, stride);


        shape::ShapeInformation shapeInfo;
        shapeInfo.shape = shape;
        shapeInfo.stride = stride;
        shapeInfo.offset = 0;
        shapeInfo.rank = rank;
        auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

        shapeInfo.order = 'f';
        shapeInfo.elementWiseStride = elementWiseStride;
        shape::toShapeBuffer(&shapeInfo, output);
        nd4j::ArrayOptions::setDataType(output, dtype);
        return output;
    }

/**
 * Compute the real linear indices for the given shape and stride
 */
    INLINEDEF _CUDA_HD Nd4jLong *computeIndices(int rank, Nd4jLong *shape,  Nd4jLong *stride) {
        Nd4jLong length = shape::prodLong(shape,rank);

        traceNew(13);

        Nd4jLong *ret = new Nd4jLong[length];
        for(int i = 0; i < length; i++) {
            Nd4jLong *idx = shape::ind2sub(rank, shape, i);
            ret[i] = shape::getOffset(0, shape, stride, idx, rank);
            delete[] idx;
        }

        return ret;
    }

/**
* Compute the real linear indices for the given shape and stride
*/
    INLINEDEF _CUDA_HD Nd4jLong *computeIndices(Nd4jLong *shapeBuffer) {
        return computeIndices(shape::rank(shapeBuffer),shape::shapeOf(shapeBuffer),shape::stride(shapeBuffer));
    }


/**
* Convert the given index (such as 1,1)
* to a linear index
* @param shape the shape of the indexes to convert
* @param indices the index to convert
* @return the linear index given the shape
* and indices
*/
    INLINEDEF _CUDA_HD int sub2Ind(int rank, Nd4jLong *shape, Nd4jLong *indices) {
        int index = 0;
        int shift = 1;

        for(int i = 0; i < rank; i++) {
            index += shift * indices[i];
            shift *= shape[i];
        }
        return index;
    }


template <typename T>
 INLINEDEF _CUDA_HD void fill(T* buffer, T value, Nd4jLong length) {

#pragma omp simd
     for (int e = 0; e < length; e++)
        buffer[e] = value;
 }

/**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong* ind2sub(int rank,  Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices) {
        auto ret = new Nd4jLong[rank];
        ind2sub(rank, shape, index, numIndices, ret);
        return ret;
    }

/**
 * Convert a linear index to
 * the equivalent nd index.
 * Infers the number of indices from the specified shape.
 *
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @return the mapped indexes along each dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong* ind2sub(int rank,  Nd4jLong *shape, Nd4jLong index) {
        return ind2sub(rank,shape, index, shape::prodLong(shape,rank));
    }

/**
* Convert a linear index to
* the equivalent nd index
* @param shape the shape of the dimensions
* @param index the index to map
* @param numIndices the number of total indices (typically prod of shape(
* @return the mapped indexes along each dimension
*/
    INLINEDEF _CUDA_HD void ind2sub(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices, Nd4jLong *ret) {
        int denom = numIndices;

        for(int i = rank - 1; i >= 0; i--) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;
        }
    }

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    INLINEDEF _CUDA_HD void ind2sub(int rank,Nd4jLong *shape, Nd4jLong index, Nd4jLong *out) {
        ind2sub(rank,shape, index, shape::prodLong(shape,rank),out);
    }

/**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong * ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices) {
        auto ret = new Nd4jLong[rank];
        ind2subC(rank, shape, index, numIndices, ret);
        return ret;
    }

/**
 * Convert a linear index to
 * the equivalent nd index.
 * Infers the number of indices from the specified shape.
 *
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @return the mapped indexes along each dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong *ind2subC(int rank, Nd4jLong *shape, Nd4jLong index) {
        return ind2subC(rank,shape, index, shape::prodLong(shape,rank));
    }

/**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
    INLINEDEF _CUDA_HD void ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong numIndices, Nd4jLong *ret) {
        auto denom = numIndices;
        for(int i = 0; i < rank; i++) {
            denom /= shape[i];
            if(denom > 0) {
                ret[i] = index / denom;
                index %= denom;
            }
            else
                ret[i] = 0;
        }
    }

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    INLINEDEF _CUDA_HD void ind2subC(int rank, Nd4jLong *shape, Nd4jLong index, Nd4jLong *out) {
        ind2subC(rank,shape, index,shape::prodLong(shape,rank),out);
    }

/**
* Convert a linear index to
* the equivalent nd index
* @param shape the shape of the dimensions
* @param index the index to map
* @param numIndices the number of total indices (typically prod of shape(
* @return the mapped indexes along each dimension
*/
    INLINEDEF _CUDA_HD void ind2subOrder(Nd4jLong *shapeInfo, Nd4jLong index, Nd4jLong numIndices, Nd4jLong *out) {
        if(shape::order(shapeInfo) == 'f') {
            shape::ind2sub(
                    shape::rank(shapeInfo),
                    shape::shapeOf(shapeInfo),
                    index,
                    numIndices,
                    out);
        }
        else {
            shape::ind2subC(
                    shape::rank(shapeInfo),
                    shape::shapeOf(shapeInfo),
                    index,
                    numIndices,
                    out);

        }
    }

/**
* Convert a linear index to
* the equivalent nd index
* @param shape the shape of the dimensions
* @param index the index to map
* @param numIndices the number of total indices (typically prod of shape(
* @return the mapped indexes along each dimension
*/
    INLINEDEF _CUDA_HD void ind2subOrder(Nd4jLong *shapeInfo, Nd4jLong index, Nd4jLong *out) {
        ind2subOrder(shapeInfo,index,shape::length(shapeInfo),out);
    }

/**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */



/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
    INLINEDEF _CUDA_HD Nd4jLong *doPermuteSwap(int length, Nd4jLong *shape,  int *rearrange) {
        traceNew(16);
        Nd4jLong *ret = new Nd4jLong[length];
        for (int i = 0; i < length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
    INLINEDEF _CUDA_HD void doPermuteSwap(int length, Nd4jLong **shape, int *rearrange) {
        if(length == 1) {
            return;
        }
        else {
            Nd4jLong *shapeDeref = *shape;
            if(shape::prodLong(shapeDeref,length) < 2) {
                return;
            }
        }

        bool inOrder = true;
        for(int i = 0; i < length - 1; i++) {
            inOrder = inOrder && rearrange[i] + 1 == rearrange[i + 1];

        }

        //all in order, nothing to do
        if(inOrder)
            return;


        Nd4jLong *shapeDeref = *shape;
        //we know they are just reversed, dimension length of 2
        if(length == 2) {
            auto shapeFirst = shapeDeref[0];
            auto shapeSecond = shapeDeref[1];
            shapeDeref[0] = shapeSecond;
            shapeDeref[1] = shapeFirst;
            return;
        }
        else if(length == 1) {
            //no permute
            return;
        }

        auto temp = new Nd4jLong[length];
        memcpy(temp,shapeDeref,sizeof(Nd4jLong) * length);
        for (int i = 0; i < length; i++) {
            shapeDeref[i] = temp[rearrange[i]];
        }

        delete[] temp;
    }


    INLINEDEF _CUDA_HD void permuteShapeBufferInPlace(Nd4jLong *shapeBuffer, int *rearrange, Nd4jLong *out) {
        if(shapeBuffer != out)
            memcpy(out,shapeBuffer,sizeof(Nd4jLong) * shape::shapeInfoLength(shape::rank(shapeBuffer)));

        doPermuteShapeBuffer(shape::rank(shapeBuffer), shapeBuffer, rearrange, out);
    }

    INLINEDEF _CUDA_HD Nd4jLong *permuteShapeBuffer(Nd4jLong *shapeBuffer, int* rearrange) {
        auto len = shape::shapeInfoLength(shape::rank(shapeBuffer));
        Nd4jLong *copy = shape::copyOf(len, shapeBuffer);
        doPermuteShapeBuffer(copy,rearrange);
        return copy;
    }

    INLINEDEF _CUDA_HD void doPermuteShapeInfo(Nd4jLong *shapeInfo, const Nd4jLong *rearrange) {

        const int rank = shape::rank(shapeInfo);

        //check whether shape is like {1} or {1,1} or {1,1,1,1,...} - in this case we don't need permute
        if(prodLong(shape::shapeOf(shapeInfo), rank) < 2)
            return;

        // check whether rearrange is like {0,1,2,3,...}  - in this case we don't need permute as well
        bool isPermutNecessary = false;
        for(int i = 0; i < rank; ++i)
            if(rearrange[i] != i) {
                isPermutNecessary = true;
                break;
            }

        if(!isPermutNecessary)
            return;

        // check whether rearrange contains correct indexes
        for(int i = 0; i < rank; ++i)
            if(rearrange[i] >= rank || rearrange[i] < 0) {
                printf("shape::doPermuteShapeInfo function failed: rearrange indexes are incorrect !\n");
                return;
            }

        // if everything is ok then perform permute
        auto temp = new Nd4jLong[shape::shapeInfoLength(rank)];
        memcpy(temp, shapeInfo, sizeof(Nd4jLong) * shape::shapeInfoLength(rank));
        for (int i = 0; i < rank; ++i) {
            shapeInfo[i + 1]        = temp[rearrange[i] + 1];
            shapeInfo[i + 1 + rank] = temp[rearrange[i] + 1 + rank];
        }

        shapeInfo[shapeInfoLength(rank) - 2] = -1;
        shapeInfo[shape::shapeInfoLength(rank) - 1] = shape::getOrder(rank, shape::shapeOf(shapeInfo),shape::stride(shapeInfo),1);

        delete[] temp;
    }

    INLINEDEF _CUDA_HD void doPermuteShapeInfo(Nd4jLong *shapeInfo, const int* rearrange) {

        const int rank = shape::rank(shapeInfo);
        
        //check whether shape is like {1} or {1,1} or {1,1,1,1,...} - in this case we don't need permute 
        if(prodLong(shape::shapeOf(shapeInfo), rank) < 2)
            return;
        
        // check whether rearrange is like {0,1,2,3,...}  - in this case we don't need permute as well 
        bool isPermutNecessary = false;
        for(int i = 0; i < rank; ++i)
            if(rearrange[i] != i) {
               isPermutNecessary = true;
               break;  
            }
        
        if(!isPermutNecessary)
            return;

        // check whether rearrange contains correct indexes
        for(int i = 0; i < rank; ++i)
            if(rearrange[i] >= rank || rearrange[i] < 0) {
                printf("shape::doPermuteShapeInfo function failed: rearrange indexes are incorrect !\n");
                return;
            }

        // if everything is ok then perform permute 
        auto temp = new Nd4jLong[shape::shapeInfoLength(rank)];
        memcpy(temp, shapeInfo, sizeof(Nd4jLong) * shape::shapeInfoLength(rank));
        for (int i = 0; i < rank; ++i) {
            shapeInfo[i + 1]        = temp[rearrange[i] + 1];
            shapeInfo[i + 1 + rank] = temp[rearrange[i] + 1 + rank];
        }

        shapeInfo[shapeInfoLength(rank) - 2] = -1;
        shapeInfo[shape::shapeInfoLength(rank) - 1] = shape::getOrder(rank, shape::shapeOf(shapeInfo),shape::stride(shapeInfo),1);

        delete[] temp;
    }

    INLINEDEF _CUDA_HD void doPermuteShapeBuffer(Nd4jLong *shapeBuffer,int *rearrange) {

        //no swapping needs to happen
        if(shape::isScalar(shapeBuffer)) {
            return;
        }

        Nd4jLong *shapeRef = shapeBuffer;
        //rank of the rearrange array == rank of shape buffer
        int rearrageRank = shape::rank(shapeRef);
        Nd4jLong *shape = shape::shapeOf(shapeRef);
        Nd4jLong *stride = shape::stride(shapeRef);
        shape::doPermuteSwap(rearrageRank,&shape,rearrange);
        shape::doPermuteSwap(rearrageRank,&stride,rearrange);
        shapeRef[shapeInfoLength(rearrageRank) - 2] = -1;
        shapeRef[shape::shapeInfoLength(rearrageRank) - 1] = shape::getOrder(rearrageRank,shape,stride,1);

        // doPermuteShapeInfo(shapeBuffer, rearrange); // possible fix of integer overflow issue when strides are too large
    }
/*
    INLINEDEF _CUDA_HD void doPermuteShapeBuffer(Nd4jLong *shapeBuffer, int *rearrange, Nd4jLong *tmpBuffer) {
        auto shapeRef = shapeBuffer;
        //rank of the rearrange array == rank of shape buffer
        int rearrageRank = shape::rank(shapeRef);
        auto shape = shape::shapeOf(shapeRef);
        auto stride = shape::stride(shapeRef);

        shape::copyOf(rearrageRank,rearrange, tmpBuffer);
        shape::doPermuteSwap(rearrageRank,&shape, tmpBuffer);

        shape::copyOf(rearrageRank,rearrange, tmpBuffer);
        shape::doPermuteSwap(rearrageRank,&stride,tmpBuffer);

        shapeRef[shapeInfoLength(rearrageRank) - 2] = -1;
        shapeRef[shape::shapeInfoLength(rearrageRank) - 1] = shape::getOrder(rearrageRank,shape,stride,1);
    }
    */

    INLINEDEF _CUDA_HD void doPermuteShapeBuffer(int rank,Nd4jLong *shapeBuffer, int *rearrange) {
        Nd4jLong *shapeRef = shapeBuffer;
        //rank of the rearrange array == rank of shape buffer
        int rearrageRank = rank;
        Nd4jLong *shape = shape::shapeOf(shapeRef);
        Nd4jLong *stride = shape::stride(shapeRef);
        auto rearrangeCopy1 = shape::copyOf(rearrageRank, rearrange);
        shape::doPermuteSwap(rearrageRank,&shape,rearrangeCopy1);
        delete[] rearrangeCopy1;
        auto rearrangeCopy2 = shape::copyOf(rearrageRank,rearrange);
        shape::doPermuteSwap(rearrageRank, &stride, rearrangeCopy2);
        shapeBuffer[shape::shapeInfoLength(rank) - 1] = shape::getOrder(rank,shape,stride,1);
        shapeBuffer[shape::shapeInfoLength(rank) - 2] = -1;
        delete[] rearrangeCopy2;
    }

    INLINEDEF _CUDA_HD void doPermuteShapeBuffer(int rank, Nd4jLong *shapeBuffer, int *rearrange, Nd4jLong *tmpBuffer) {
        Nd4jLong *shapeRef = shapeBuffer;
        //rank of the rearrange array == rank of shape buffer
        int rearrageRank = rank;
        auto shape = shape::shapeOf(shapeRef);
        auto stride = shape::stride(shapeRef);
        if(shapeBuffer != tmpBuffer)
            shape::copyOf(rearrageRank,shapeBuffer, tmpBuffer);

        shape::doPermuteSwap(rearrageRank,&shape,rearrange);
        shape::doPermuteSwap(rearrageRank,&stride,rearrange);
        shapeRef[shapeInfoLength(rank) - 2] = -1;
        shapeRef[shape::shapeInfoLength(rank) - 1] = shape::getOrder(rank,shape,stride,1);
    }


    INLINEDEF _CUDA_HD Nd4jLong *createPermuteIndexes(int originalRank, int *dimension,int dimensionLength) {
        int delta = originalRank - dimensionLength;

        traceNew(17);

        Nd4jLong *ret = new Nd4jLong[originalRank];
        for(int i = 0; i < delta; i++) {
            ret[i] = i + dimensionLength;
        }

        for(int i = delta; i  < originalRank; i++) {
            ret[i] = i - delta;
        }

        return ret;
    }

/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
    INLINEDEF _CUDA_HD char getOrder(int length, Nd4jLong *shape, Nd4jLong *stride, int elementStride) {
        int sd = -1;
        int dim = -1;
        int i = -1;
        int cContiguous = 1;
        int isFortran = 1;

        sd = 1;
        for (i = length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = 0;
                break;
            }
            /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = 0;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        if (isFortran && cContiguous)
            return 'a';
        else if (isFortran && !cContiguous)
            return 'f';
        else if (!isFortran && !cContiguous)
            return 'c';
        else
            return 'c';

    }





/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */

    template <typename T>
    INLINEDEF _CUDA_HD int checkArrangeArray(T *arr, int arrLength, int shapeLength) {
        if (arrLength != shapeLength)
            return -1;
        for (int i = 0; i < arrLength; i++) {
            if (arr[i] >= arrLength || arr[i] < 0)
                return -1;
        }

        for (int i = 0; i < arrLength; i++) {
            for (int j = 0; j < arrLength; j++) {
                if (i != j && arr[i] == arr[j])
                    return -1;
            }
        }

        return 1;
    }


    INLINEDEF _CUDA_HD void traceNew(int id) {
        //printf("new happened: [%i]\n", id);

#ifndef __CUDACC__
        //fflush(stdout);
#endif
    }

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
    INLINEDEF _CUDA_HD void permute(ShapeInformation **info, int *rearrange, int rank) {
        ShapeInformation *infoDeref = *info;
        checkArrangeArray(rearrange, rank, rank);
        shape::doPermuteSwap(rank, &infoDeref->shape, rearrange);
        shape::doPermuteSwap(rank, &infoDeref->stride, rearrange);
        char order = getOrder(rank,
                              infoDeref->shape,
                              infoDeref->stride,
                              infoDeref->elementWiseStride);
        infoDeref->order = order;

    }

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
    INLINEDEF _CUDA_HD int isVector(Nd4jLong *shape, int rank) {
        if (rank == 0)
            return 0;

        if (rank == 1)
            return 1;

        if (rank > 2)
            return 0;
        else if (rank <= 2) {
            if (shape[0] == 1 || shape[1] == 1)
                return 1;
        }
        return 0;
    }

    INLINEDEF _CUDA_HD bool isLikeVector(Nd4jLong *shapeInfo, int& posOfNonUnityDim) {
                
        int numOfNonUnity = 0;
        for(int i = 1; i <= shapeInfo[0]; ++i) {
            if(shapeInfo[i] != 1) {
                ++numOfNonUnity;
                posOfNonUnityDim = i-1;
            }
        }
        
        return numOfNonUnity == 1 && shapeInfo[0] > 2;
    }

    INLINEDEF _CUDA_H Nd4jLong* detachShape(Nd4jLong *originalShape) {
        Nd4jLong *newShape = new Nd4jLong[shape::shapeInfoLength(originalShape)];
        memcpy(newShape, originalShape, shape::shapeInfoByteLength(originalShape));

        return newShape;
    }


    INLINEDEF _CUDA_H Nd4jLong* copyShape(Nd4jLong *originalShape) {
        Nd4jLong *newShape = new Nd4jLong[shape::shapeInfoLength(originalShape)];
        memcpy(newShape, originalShape, shape::shapeInfoByteLength(originalShape));

        return newShape;
    }

    INLINEDEF _CUDA_HD int isVector(Nd4jLong *shapeInfo) {
        return isVector(shape::shapeOf(shapeInfo),shape::rank(shapeInfo));
    }

    INLINEDEF _CUDA_HD bool isRowVector(Nd4jLong *shapeInfo) {
        bool isVector = shape::isVector(shapeInfo) == 1;
        bool shapeFirstOne = shape::shapeOf(shapeInfo)[0] == 1;
        return isVector && shapeFirstOne;
    }

    INLINEDEF _CUDA_HD bool isColumnVector(Nd4jLong *shapeInfo) {
        bool isVector = shape::isVector(shapeInfo) == 1;
        bool shapeFirstOne = shape::shapeOf(shapeInfo)[0] == 1;
        return isVector && !shapeFirstOne;
    }

    INLINEDEF _CUDA_HD int oneDimEqualToLength(Nd4jLong *shape, int rank) {
        for(int i = 0; i < rank; i++) {
            if(shape[i] == shape::prod(shape,rank))
                return 1;
        }

        return 0;
    }

    INLINEDEF _CUDA_HD int oneDimEqualToLength(Nd4jLong *shapeInfo) {
        return oneDimEqualToLength(shape::shapeOf(shapeInfo),shape::rank(shapeInfo));
    }

/**
* Returns whether the
* given shape is a vector or not
* @param shape the shape of the array
* @param rank the rank of the shape
*/
    INLINEDEF _CUDA_HD int isMatrix(Nd4jLong *shape, int rank) {
        if (rank > 2)
            return 0;
        else if (rank <= 2) {
            if (shape[0] == 1 || shape[1] == 1)
                return 0;
        }

        return 1;
    }

    INLINEDEF _CUDA_HD int isMatrix(Nd4jLong *shapeInfo) {
        return isMatrix(shape::shapeOf(shapeInfo),shape::rank(shapeInfo));
    }

/**
 * Returns the shape portion of an information
 * buffer
 */
    INLINEDEF _CUDA_HD Nd4jLong *shapeOf(Nd4jLong *buffer) {
        return buffer + 1;
    }

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
    template <typename T>
    INLINEDEF _CUDA_HD T *copyOf(Nd4jLong length, T *toCopy) {
        traceNew(18);

        T *ret = new T[length];
        return copyOf(length, toCopy, ret);
    }

    template <typename T>
    INLINEDEF _CUDA_HD T* copyOf(Nd4jLong length, T *toCopy, T *ret) {
        memcpy(ret, toCopy, sizeof(T)*length);
        return ret;
    }

/**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
    template <typename T>
    INLINEDEF _CUDA_HD void copyTo(Nd4jLong length, T *from, T *to) {
        memcpy(to, from, sizeof(T)*length);
    }

/**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
    INLINEDEF _CUDA_HD void copyTo(int length, Nd4jLong *from, Nd4jLong *to, Nd4jLong *indexes) {
        for(int i = 0; i < length; i++) {
            to[i] = from[indexes[i]];
        }
    }

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
 /*
    INLINEDEF _CUDA_HD Nd4jLong *permutedStrides(Nd4jLong *toPermute, int shapeRank, int *rearrange) {
        Nd4jLong *strideCopy = copyOf(shapeRank, toPermute);
        checkArrangeArray(rearrange, shapeRank, shapeRank);
        Nd4jLong *newStride = doPermuteSwap(shapeRank, strideCopy, rearrange);
        delete[] strideCopy;
        return newStride;
    }
    */

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
    INLINEDEF _CUDA_HD Nd4jLong *slice(Nd4jLong *shape) {
        return shape + 1;
    }

    INLINEDEF _CUDA_HD int slices(Nd4jLong *shapeBuffer) {
        return static_cast<int>(shape::shapeOf(shapeBuffer)[0]);
    }


    INLINEDEF _CUDA_HD Nd4jLong *sliceOfShapeBuffer(Nd4jLong sliceIdx, Nd4jLong *shapeBuffer) {
        int rank = shape::rank(shapeBuffer);
        int newRank = rank - 1;
        if(newRank < 2)
            newRank = 2;
        Nd4jLong *newShapeBuffer = new Nd4jLong[shape::shapeInfoLength(newRank)];
        newShapeBuffer[0] = newRank;
        Nd4jLong *currShape = shape::shapeOf(shapeBuffer);
        Nd4jLong *currStride = shape::stride(shapeBuffer);
        //initialize new shape and stride by taking the shape and stride + 1
        //and adding to the shape information
        //a slice is always just taking the existing shape and cutting the first index off
        //of the shape and stride
        Nd4jLong *newShape = shape::shapeOf(newShapeBuffer);
        Nd4jLong *newStride = shape::stride(newShapeBuffer);
        if(shape::isVector(shapeBuffer)) {
            Nd4jLong *currShape = shape::shapeOf(shapeBuffer);
            //row vector: slice index 0 is a valid index, just copy the whole thing
            if(currShape[0] == 1) {
                if(sliceIdx == 0) {
                    memcpy(newShapeBuffer,shapeBuffer,shape::shapeInfoByteLength(shape::rank(shapeBuffer)));
                    return newShapeBuffer;
                }
            }
                //column vector: this will be a scalar
            else {
                delete[] newShapeBuffer;
                Nd4jLong *scalar = shape::createScalarShapeInfo();
                int offset = shape::offset(shapeBuffer);
                scalar[shape::shapeInfoLength(2) - 3] = offset + sliceIdx;
                return scalar;
            }
        }
        else if(shape::isMatrix(shapeBuffer)) {
            newShape[0] = 1;
            newShape[1] = currShape[1];
            newStride[0] = 1;
            newStride[1] = currStride[1];
        }
        else {
            for(int i = 0; i < newRank; i++) {
                newShape[i] = currShape[i + 1];
                newStride[i] = currStride[i + 1];
            }
        }

        auto indices = new Nd4jLong[rank];
        memset((void *) indices,0,rank * sizeof(Nd4jLong));
        indices[0] = sliceIdx;
        Nd4jLong offset = shape::getOffset(0,newShape,newStride,indices,rank);
        newShapeBuffer[shape::shapeInfoLength(newRank) - 3] = offset;
        if(shape::isMatrix(shapeBuffer)) {
            newShapeBuffer[shape::shapeInfoLength(newRank) - 2] = currStride[1];
        }
        else {
            newShapeBuffer[shape::shapeInfoLength(newRank) - 2] = shape::elementWiseStride(shapeBuffer);
        }
        newShapeBuffer[shape::shapeInfoLength(newRank) - 1] = shape::getOrder(newRank,newShape,newStride,1);


        delete[] indices;

        return newShapeBuffer;
    }

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
    INLINEDEF _CUDA_HD int shapeInfoLength(int rank) {
        //FIXME magic numbers
        return rank * 2 + 4;
    }

    INLINEDEF _CUDA_HD int shapeInfoLength(Nd4jLong* shape) {
        return shapeInfoLength(static_cast<int>(shape[0]));
    }

    INLINEDEF _CUDA_HD int shapeInfoLength(const Nd4jLong* shape) {
        return shapeInfoLength(static_cast<int>(shape[0]));
    }

    INLINEDEF _CUDA_HD size_t shapeInfoByteLength(int rank) {
        //FIXME magic numbers
        return (rank * 2 + 4) * sizeof(Nd4jLong);
    }

    INLINEDEF _CUDA_HD size_t shapeInfoByteLength(Nd4jLong* shapeInfo) {
        //FIXME magic numbers
        return shapeInfoByteLength((int) shapeInfo[0]);
    }

    INLINEDEF _CUDA_HD size_t shapeInfoByteLength(const Nd4jLong* shapeInfo) {
        //FIXME magic numbers
        return shapeInfoByteLength((int) shapeInfo[0]);
    }

/**
 * Returns the rank portion of
 * an information buffer
 */
    INLINEDEF _CUDA_HD  int rank(const Nd4jLong *buffer) {
        return static_cast<int>(buffer[0]);
    }

/**
 * Converts a raw int buffer of the layout:
 * rank
 * shape
 * stride
 * offset
 * elementWiseStride
 *
 * where shape and stride are both straight int pointers
 */
    INLINEDEF _CUDA_HD ShapeInformation *infoFromBuffer(Nd4jLong *buffer) {

        traceNew(19);

        auto info = new ShapeInformation;
        auto length = shapeInfoLength(rank(buffer));
        auto rank = buffer[0];

        //start after rank
        info->shape = buffer + 1;
        info->stride = buffer + (1 + rank);
        info->rank = rank;
        info->offset = buffer[length - 3];
        info->elementWiseStride = buffer[length - 2];
        Nd4jLong *stride = buffer + 1 + rank;
        info->stride = stride;
        info->order = (char) buffer[length - 1];
        return info;
    }

/**
 * Returns the stride portion of an information
 * buffer
 */
    INLINEDEF _CUDA_HD Nd4jLong *stride( Nd4jLong *buffer) {
        return buffer + (1 + rank(buffer));
    }

    INLINEDEF _CUDA_HD bool isEmpty(Nd4jLong *shapeInfo) {
        return ((shape::extra(shapeInfo) & ARRAY_EMPTY) == ARRAY_EMPTY);
    }


/**
 * Compute the length of the given shape
 */
    INLINEDEF _CUDA_HD Nd4jLong length(Nd4jLong *shapeInfo) {
        int rank = shape::rank(shapeInfo);
        if (rank == 0) {
            if (isEmpty(shapeInfo))
                return 0L;
            else
                return 1L;
        }
        if (rank == 1)
            return shapeInfo[1];


        return shape::prodLong(shape::shapeOf(shapeInfo), rank);
    }

    INLINEDEF _CUDA_HD Nd4jLong length(std::initializer_list<int>& shape) {
        Nd4jLong ret = 1;
        for (auto v : shape) {
            ret *= v;
        }
        return ret;
    }

    INLINEDEF _CUDA_HD Nd4jLong length(std::initializer_list<Nd4jLong>& shape) {
        Nd4jLong ret = 1;
        for (auto v : shape) {
            ret *= v;
        }
        return ret;
    }

/***
 * Returns the offset
 * portion of an information buffer
 */
    INLINEDEF _CUDA_HD Nd4jLong offset(Nd4jLong *buffer) {
        return buffer[shape::shapeInfoLength(shape::rank(buffer)) - 3];
    }

    INLINEDEF _CUDA_HD Nd4jLong& extra(Nd4jLong *buffer) {
        return buffer[shape::shapeInfoLength(shape::rank(buffer)) - 3];
    }


/**
 * Returns the ordering
 * for this shape information buffer
 */
    INLINEDEF _CUDA_HD char order(const Nd4jLong *buffer) {
        //FIXME magic numbers
        return static_cast<char>(buffer[(buffer[0] * 2 + 4) - 1]);
    }

/**
 * Returns the element wise stride for this information
 * buffer
 */
    INLINEDEF _CUDA_HD Nd4jLong elementWiseStride(Nd4jLong *buffer) {
        return buffer[shapeInfoLength(buffer[0]) - 2];
    }

/**
* Returns the element wise stride for this information
* buffer relative to a dimension and reduction index
*/
    INLINEDEF _CUDA_HD Nd4jLong reductionIndexElementWiseStride(Nd4jLong* buffer, int* dimension, int dimensionLength) {
        if(dimensionLength > 1) {
            if(shape::order(buffer) == 'f') {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                if(shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
                    //int tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                    //return tadElementWiseStride;
                    auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
                    return tadElementWiseStride;
                }

                return 1;

            }
            else {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                if(shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
                    auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                    return tadElementWiseStride;
                }

                return 1;
            }
        }
        else {
            if(shape::order(buffer) == 'f') {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
                return tadElementWiseStride;
            }
            else {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                return tadElementWiseStride;
            }
        }

    }

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
    INLINEDEF _CUDA_HD int isScalar(Nd4jLong *info) {

        const int rank = shape::rank(info);

        if(rank > 2)
            return 0;
        if(rank == 0)
            return 1;
        if(rank == 1)
            return shape::shapeOf(info)[0] == 1;
        if(rank == 2)
            return shape::shapeOf(info)[0] == 1 && shape::shapeOf(info)[1] == 1;

        return 0;
    }

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
    INLINEDEF _CUDA_HD int isScalar(volatile ShapeInformation *info) {

        const int rank = info->rank;

        if(rank > 2)
            return 0;
        if(rank == 1)
            return info->shape[0] == 1;
        if(rank == 2)
            return info->shape[0] == 1 && info->shape[1] == 1;

        return 0;
    }

/**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
    template <typename T1, typename T2>
    INLINEDEF _CUDA_HD void removeIndex(T1* data, T2 *indexes, Nd4jLong dataLength, Nd4jLong indexesLength, T1 *ret) {

        int count = 0;
        int absLength = dataLength - indexesLength;
        for (int i = 0; i < dataLength && count < absLength; i++) {
            int contains = 0;
            for (int j = 0; j < indexesLength; j++) {
                if (i == indexes[j]) {
                    contains = 1;
                    break;
                }
            }

            if (!contains) {
                ret[count] = data[i];
                count++;
            }
        }
    }

    /**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
    template <typename T1, typename T2>
    INLINEDEF _CUDA_HD T1* removeIndex(T1 *data, T2 *indexes, Nd4jLong dataLength, Nd4jLong indexesLength) {
        auto lengthOfArr = dataLength - indexesLength;
        if(lengthOfArr < 0) {
            printf("Remove index call created a <= 0 length array. This was likely not intended.");
        }

        auto ret = new T1[lengthOfArr];
        memset(ret,0,sizeof(T1)  * lengthOfArr);
        removeIndex<T1, T2>(data, indexes, dataLength, indexesLength, ret);
        return ret;
    }

    INLINEDEF _CUDA_HD Nd4jLong* everyIndexBut(Nd4jLong *indexes,int indexesLength,int begin,int end) {
        int len = end - indexesLength;

        traceNew(20);

        auto ret = new Nd4jLong[len];
        int retIdx = 0;
        //not here that we do 0 based indexing for end - this assumes things like:
        //0 to 4 are specified
        for(int i = begin; i < end ; i++) {
            bool found = false;
            for(int j = 0; j < indexesLength; j++) {
                if(indexes[j] == i) {
                    found = true;
                    break;
                }
            }

            if(!found) {
                ret[retIdx++] = i;
            }

        }

        return ret;

    }

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
    INLINEDEF  __device__ int tadOffset(ShapeInformation *xInfo, int offset) {
    return offset + threadIdx.x * xInfo->elementWiseStride;
}
#endif

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
    INLINEDEF _CUDA_HD Nd4jLong *ensureVectorShape(Nd4jLong *shape, int dimension) {
        traceNew(21);

        Nd4jLong *ret = new Nd4jLong[2];

        if (dimension == 0) {
            ret[0] = 1;
            ret[1] = shape[0];
        } else {
            ret[0] = shape[0];
            ret[1] = 1;
        }

        return ret;
    }

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
    INLINEDEF _CUDA_HD Nd4jLong *ensureVectorShape(Nd4jLong *shape) {
        return ensureVectorShape(shape, 0);
    }

    /**
     * This method does STRICT comparison for two shape buffers
     *
     * @param shape
     * @return
     */
    INLINEDEF _CUDA_HD bool equalsStrict(Nd4jLong *shapeA, Nd4jLong *shapeB) {
        if (shapeA[0] != shapeB[0])
            return false;

        if (shapeA[0] == 0)
            return true;

        // we do full comparison here
        int length = shape::shapeInfoLength(shapeA[0]);

        for (int e = 1; e < length; e++)
            if (shapeA[e] != shapeB[e])
                return false;

        return true;
    }

    INLINEDEF _CUDA_HD int sizeAt(const Nd4jLong *shape, const int dim) {
        if (dim >= 0)
            return shape[1+dim];
        else
            return shape[1+(rank(shape) + dim)];
    }

    /**
     * This method does SOFT comparison for two shape buffers, we compare only rank & shapes
     *
     * @param shape
     * @return
     */
    INLINEDEF _CUDA_HD bool equalsSoft(Nd4jLong *shapeA, Nd4jLong *shapeB) {
        if (shapeA[0] != shapeB[0])
            return false;

        if (shapeA[0] == 0)
            return true;

        // we compare only shapes, and ignoring stride & ews
        auto length = shapeA[0];

        for (int e = 1; e <= length; e++)
            if (shapeA[e] != shapeB[e])
                return false;

        return true;
    }

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
    template <typename T>
    INLINEDEF _CUDA_HD T* range(int from, int to, int increment) {
        int diff = nd4j::math::nd4j_abs<int>(from - to);
        int retLength = diff / increment;
        T *ret;

        traceNew(22);

        if(diff / increment < 1)
            ret = new T[1];
        else
            ret = new T[diff / increment];
        if (from < to) {
            int count = 0;
            for (int i = from; i < to; i += increment) {
                if (count >= retLength)
                    break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = from - 1; i >= to; i -= increment) {
                if (count >= retLength)
                    break;
                ret[count++] = i;
            }
        }

        return ret;
    }

/**
 * Generate a range
 * beginning at from and ending at to
 * incrementing by 1
 * @param from the start
 * @param to the end
 * @return the int array starting at from and ending at to
 */

    template <typename T>
    INLINEDEF _CUDA_HD T* range(int from, int to) {
        return range<T>(from, to, 1);
    }

/**
 * Keep the given indexes in the data
 * @param data
 * @param index
 * @param indexLength
 * @param dataLength
 * @return
 */
    INLINEDEF _CUDA_HD Nd4jLong *keep(volatile Nd4jLong *data, int* index, int indexLength, int dataLength) {

        traceNew(23);

        Nd4jLong *ret = new Nd4jLong[indexLength];
        int count = 0;
        for (int i = 0; i < dataLength; i++) {
            int contains = 0;
            for (int j = 0; j < indexLength; j++) {
                if (i == index[j]) {
                    contains = 1;
                    break;
                }
            }

            if (contains)
                ret[count++] = data[i];
        }
        return ret;
    }

/**
 * Generate a reverse
 * copy of the data
 */

    template <typename T>
    INLINEDEF _CUDA_HD T* reverseCopy(T *data, Nd4jLong length) {
        if (length < 1)
            return nullptr;

        traceNew(24);

        T *copy = new T[length];
        for (Nd4jLong i = 0; i <= length / 2; i++) {
            T temp = data[i];
            copy[i] = data[length - i - 1];
            copy[length - i - 1] = temp;
        }
        return copy;
    }

    template <typename T>
    INLINEDEF _CUDA_HD void reverseCopyTo(T *from, T *to, Nd4jLong length) {
        if (length < 1)
            return;
        for (Nd4jLong i = 0; i <= length / 2; i++) {
            T temp = from[i];
            to[i] = from[length - i - 1];
            to[length - i - 1] = temp;
        }
    }

    template <typename T>
    INLINEDEF _CUDA_HD void reverseCopyTo(T *from, T *to, Nd4jLong *indexes, Nd4jLong length) {
        if (length < 1)
            return;

        for (Nd4jLong i = 0; i <= length / 2; i++) {
            T temp = from[indexes[i]];
            to[i] = from[indexes[length - i - 1]];
            to[length - i - 1] = temp;
        }

    }

/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
    template <typename T>
    INLINEDEF _CUDA_HD T* concat(T* arr1, Nd4jLong arr1Length, T* arr2, Nd4jLong arr2Length) {

        traceNew(25);

        T *ret = new T[arr1Length + arr2Length];
        std::memcpy(ret, arr1, arr1Length * sizeof(T));
        std::memcpy(ret + arr1Length, arr2, arr2Length * sizeof(T));
        return ret;
    }

/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
    template <typename T>
    INLINEDEF _CUDA_HD T *concat(Nd4jLong numArrays, Nd4jLong numTotalElements, T **arr, Nd4jLong *lengths) {

        T* ret = new T[numTotalElements];
        Nd4jLong count = 0;

        for (Nd4jLong i = 0; i < numArrays; i++) {
            for (Nd4jLong j = 0; j < lengths[i]; j++) {
                ret[count++] = arr[i][j];
            }
        }

        return ret;
    }

/**
 * Get the length per slice of the
 * given shape and the dimension
 * @param rank the rank of the shape
 * @param shape the shape of to get
 * the length per slice for
 * @param dimension the dimension to
 * get the length per slice for
 * @param dimensionLength the length of the dimension array
 * @return the length per slice of the given shape
 * along the given dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong lengthPerSlice(int rank, Nd4jLong *shape, int* dimension, int dimensionLength) {
        if(shape::isVector(shape,rank)) {
            //return total length for row vectors
            if(dimensionLength == 1 && shape[0] == 1) {
                return shape::prod(shape,rank);
            }
        }
        else if(rank == dimensionLength)
            return shape::prod(shape,rank);
        int absSelta = nd4j::math::nd4j_abs<int>(rank - dimensionLength);
        traceNew(27);
        auto ret2 = shape::removeIndex<Nd4jLong>(shape, dimension, rank, dimensionLength);
        auto ret = prodLong(ret2, absSelta);
        delete[] ret2;
        return ret;
    }

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    INLINEDEF _CUDA_HD Nd4jLong sliceOffsetForTensor(int rank, int index, Nd4jLong *shape, Nd4jLong *tensorShape, int tensorShapeLength, int* dimension, int dimensionLength) {
        auto tensorLength = prodLong(tensorShape, tensorShapeLength);
        auto lengthPerSlice2 = lengthPerSlice(rank, shape, dimension, dimensionLength);
        if (lengthPerSlice2 <= 0) {
            return 0;
        }

        Nd4jLong offset = index * tensorLength / lengthPerSlice2;
        return offset;
    }

    /**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */

    INLINEDEF _CUDA_HD Nd4jLong sliceOffsetForTensor(int index,int tensorLength,int lengthPerSlice2) {
        Nd4jLong offset = index * tensorLength / lengthPerSlice2;
        return offset;
    }


#ifdef __CUDACC__
/**
* Computes the offset for accessing
* a global element given the shape information
* and the offset to be read.
*/
    INLINEDEF _CUDA_D int tadOffset(Nd4jLong *xInfo, int offset) {
    return offset + threadIdx.x * elementWiseStride(xInfo);

}
#endif





/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong tensorsAlongDimension(volatile int rank, volatile int length,
                                        volatile Nd4jLong *shape, int *dimension, int dimensionLength) {
        Nd4jLong *tensorShape = shape::keep(shape, dimension, dimensionLength, rank);
        Nd4jLong ret = length / shape::prodLong(tensorShape, dimensionLength);
        delete[] tensorShape;
        return ret;
    }

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    INLINEDEF _CUDA_HD Nd4jLong tensorsAlongDimension(Nd4jLong *shapeInfo, int *dimension, int dimensionLength) {
        Nd4jLong *keepShape = shape::shapeOf(shapeInfo);
        Nd4jLong *tensorShape = shape::keep(keepShape, dimension, dimensionLength, rank(shapeInfo));
        Nd4jLong ret = shape::length(shapeInfo) / shape::prodLong(tensorShape, dimensionLength);
        delete[] tensorShape;
        return ret;
    }




/**
* Get an offset for retrieval
* from a data buffer
* based on the given
* shape stride and given indices
* @param baseOffset the offset to start from
* @param shape the shape of the array
* @param stride the stride of the array
* @param indices the indices to iterate over
* @return the double at the specified index
*/
    INLINEDEF _CUDA_HD Nd4jLong getOffset(Nd4jLong baseOffset,  Nd4jLong *shape,  Nd4jLong *stride,  const Nd4jLong *indices, int rank) {
        Nd4jLong offset = baseOffset;
        for(int i = 0; i < rank; i++) {
            if(indices[i] >= shape[i] && shape[i] != 1) {
#ifdef __CUDA_ARCH__
                printf("D: Index %i [%lld] must not be >= shape[%lld].\n", i,indices[i],shape[i]);
#else
                printf("H: Index %i [%lld] must not be >= shape[%lld].\n", i, (long long) indices[i], (long long) shape[i]);
#endif

#ifdef __CUDA_ARCH__
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    printShapeInfoLinear("getOffsetFailed", rank, shape, stride);
#endif
                return -1;
            }

            if(shape[i] != 1) {
                offset += indices[i] * stride[i];
            }
        }

        return offset;
    }




/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
    INLINEDEF _CUDA_HD int tadForBlockIndex(int blockSize, int blockIdx, int i) {
        return blockIdx + i * blockSize;
    }

/**
 * Computes the number of tads per block
 *
 */
    INLINEDEF _CUDA_HD int tadsPerBlock(int blockSize, int tads) {
        return  nd4j::math::nd4j_ceil<double, int>(tads / (double) blockSize);
    }

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
    INLINEDEF _CUDA_HD Nd4jLong *toShapeBuffer( ShapeInformation *info) {

        traceNew(29);

        auto ret = new Nd4jLong[shapeInfoLength(info->rank)];
        int count = 1;
        int rank = info->rank;

        ret[0] = info->rank;

        for (int i = 0; i < rank; i++) {
            ret[count++] = info->shape[i];
        }

        for (int i = 0; i < rank; i++) {
            ret[count++] = info->stride[i];
        }

        ret[count++] = info->offset;
        ret[count++] = info->elementWiseStride;
        ret[count] = info->order;

        return ret;
    }

    INLINEDEF _CUDA_HD Nd4jLong *toShapeBuffer( ShapeInformation *info, Nd4jLong* ret) {

        int count = 1;
        int rank = info->rank;

        ret[0] = info->rank;

        if (ret[0] == 0) {
            ret[1] = 0;
            ret[2] = 1;
            ret[3] = 99;
            return ret;
        }

        for (int i = 0; i < rank; i++) {
            ret[count++] = info->shape[i];
        }

        for (int i = 0; i < rank; i++) {
            ret[count++] = info->stride[i];
        }

        ret[count++] = info->offset;
        ret[count++] = info->elementWiseStride;
        ret[count++] = info->order;

        return ret;
    }

    INLINEDEF _CUDA_HD void printIntArray(Nd4jLong *arr,int length) {
        for(int i = 0; i < length; i++) {
            printf(" %lld ", (long long) arr[i]);
        }

        printf("\n");
    }

    INLINEDEF _CUDA_HD void printShapeInfo(Nd4jLong *shapeInfo) {
        int rank = shape::rank(shapeInfo);
        Nd4jLong *shape = shape::shapeOf(shapeInfo);
        printf("Rank %d\n",rank);
        printf("Shape:\n");
        for(int i = 0; i < rank; i++) {
            printf(" %lld ",(long long) shape[i]);
        }

        printf("\n");

        Nd4jLong *stride = shape::stride(shapeInfo);
        printf("Stride:\n");
        for(int i = 0; i < rank; i++) {
            printf(" %lld ", (long long) stride[i]);
        }

        printf("\n");

        printf("Order %c\n",shape::order(shapeInfo));
    }

    INLINEDEF _CUDA_HD void printShapeInfoLinear(Nd4jLong *shapeInfo) {
        int rank = shape::rank(shapeInfo);
        int lim = shape::shapeInfoLength(rank);
        printf("ShapeInfo: [");
        for (int i = 0; i < lim; i++) {
            printf("%lld", (long long) shapeInfo[i]);

            if (i < lim - 1) {
                printf(", ");
            }
        }
        printf("]\n");
#ifndef __CUDA_ARCH__
        fflush(stdout);
#endif
    }


    INLINEDEF _CUDA_HD void printShapeInfoLinear(const char *msg, int rank, Nd4jLong *shape, Nd4jLong *strides) {
        printf("%s : [", msg);
        for (int i = 0; i < rank; i++) {
            printf("%lld, ", (long long) shape[i]);
        }

        for (int i = 0; i < rank; i++) {
            printf("%lld", (long long) strides[i]);

            if (i < rank - 1)
                printf(", ");
        }
        printf("]\n");

#ifndef __CUDA_ARCH__
        fflush(stdout);
#endif
    }

    INLINEDEF _CUDA_HD void printShapeInfoLinear(const char *msg, Nd4jLong *shapeInfo) {
        int rank = shape::rank(shapeInfo);
        int lim = shape::shapeInfoLength(rank);
        printf("%s : [", msg);
        for (int i = 0; i < lim; i++) {
            printf("%lld", (long long) shapeInfo[i]);

            if (i < lim - 1) {
                printf(", ");
            }
        }
        printf("]\n");
#ifndef __CUDACC__
        fflush(stdout);
#endif
    }

    template <typename T>
    INLINEDEF _CUDA_HD void printArray(T *arr,int length, const char * message) {
        if (message != nullptr)
            printf("%s: [", message);
        else
            printf("Array: [");

        for (int i = 0; i < length; i ++) {
            printf("%f", (float) arr[i]);
            if (i + 1 < length) printf(", ");
        }
        printf("]\n");

#ifndef __CUDACC__
        fflush(stdout);
#endif
    }

    INLINEDEF _CUDA_HD void printArray(float *arr,int length) {
        printf("Array: [");
        for (int i = 0; i < length; i ++) {
            printf("%f", arr[i]);
            if (i + 1 < length) printf(", ");
        }
        printf("]\n");
    }
/**
 * Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
    INLINEDEF _CUDA_HD int tadIndex(int i, int elementWiseStride, int numElementsPerTad) {
        return i / (numElementsPerTad * elementWiseStride);
    }

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
    INLINEDEF _CUDA_HD int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
                                       int tadsForOriginal) {
        if (tadIndexForOriginal == 0)
            return 0;
        return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
    }


    INLINEDEF _CUDA_HD void transposeInplace(Nd4jLong *shapeBuffer) {
        int rank = shape::rank(shapeBuffer);
        Nd4jLong *shape = shape::shapeOf(shapeBuffer);
        Nd4jLong *strides = shape::stride(shapeBuffer);

        // swap shape
        for (int e = 0; e < rank / 2; e++) {
            int idx1 = rank - e - 1;
            int idx2 =  e;
            int tmp = shape[idx2];
            shape[idx2] = shape[idx1];
            shape[idx1] = tmp;
        }

        // swap strides
        for (int e = 0; e < rank / 2; e++) {
            int idx1 = rank - e - 1;
            int idx2 =  e;
            int tmp = strides[idx2];
            strides[idx2] = strides[idx1];
            strides[idx1] = tmp;
        }

        if (shape::order(shapeBuffer) == 'c')
            shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 102;
        else
            shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 99;
    }

/**
 * Tad index for linear
 * @param linearIndex
 * @param tadLength
 * @return
 */
    INLINEDEF _CUDA_HD int tadIndexForLinear(int linearIndex, int tadLength) {
        return linearIndex % tadLength;
    }

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
    INLINEDEF _CUDA_HD int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal) {
        return tadsForOriginal / tadsForReduce;
    }

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
    INLINEDEF _CUDA_HD int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
                                          int tadNum, int originalTadNum) {
        int tad = tadIndex(i, elementWiseStride, numElementsPerTad);
        return reductionIndexForTad(tad, tadNum, originalTadNum);
    }

    INLINEDEF _CUDA_HD Nd4jLong* createScalarShapeInfo() {

        traceNew(30);

        auto shape = new Nd4jLong[1];
        shape[0] = 1;
        auto stride = new Nd4jLong[1];
        stride[0] = 1;
        auto shapeInformation2 = new ShapeInformation();
        shapeInformation2->rank = 1;
        shapeInformation2->offset = 0;
        shapeInformation2->stride = stride;
        shapeInformation2->shape = shape;
        shapeInformation2->elementWiseStride = 1;
        shapeInformation2->order = 99;
        Nd4jLong *ret = shape::toShapeBuffer(shapeInformation2);
        delete shapeInformation2;
        delete[] shape;
        delete[] stride;
        return ret;
    }

    INLINEDEF _CUDA_HD Nd4jLong* createScalarShapeInfo(Nd4jLong *ret) {
        ret[0] = 2;
        ret[1] = 1;
        ret[2] = 1;
        ret[3] = 1;
        ret[4] = 1;
        ret[5] = 0;
        ret[6] = 1;
        ret[7] = 99;

        return ret;
    }

/**
 * Returns the prod of the data
 * up to the given length
 */
    INLINEDEF _CUDA_HD int prod(Nd4jLong *data, int length) {
        int prod = 1;
        for (int i = 0; i < length; i++) {
            prod *= data[i];
        }

        return prod;
    }

/**
 * Returns the prod of the data
 * up to the given length
 */
    INLINEDEF _CUDA_HD Nd4jLong prodLong( Nd4jLong *data, int length) {
        Nd4jLong prod = 1;
        for (int i = 0; i < length; i++) {
            prod *= data[i];
        }

        return prod;
    }

    INLINEDEF _CUDA_HD int rearMostLeftOverItem(Nd4jLong *data, Nd4jLong *dimension,int dimensionLength) {
        Nd4jLong *stride = shape::stride(data);
        //corner case: return the final item when its greater than the max, since its guaranteed to be left over
        //note here that strides are interpreted in reverse for tad
        //start from the front rather than the back

        int rank = shape::rank(data);


        if(shape::order(data) == 'f') {
            int dimIdx = dimensionLength - 1;
            for(int i = rank - 1; i >= 0; i--) {
                /**
                 * Needs to find an algorithm such that:
                 * looping backwards will find the highest dimension left
                 * that isn't included in the dimension index list.
                 *
                 * This can also be thought of as the last item of the first index
                 * of the difference between the full list of indices and
                 * the dimension indices.
                 *
                 * We should avoid excessive object creation by only looping backwards.
                 */
                if(dimension[dimIdx--] != i) {
                    int ret = stride[i];
                    return ret;
                }
            }
        }

        else {
            int dimIdx = dimensionLength - 1;

            for(int i = rank - 1; i >= 0; i--) {
                /**
                 * Needs to find an algorithm such that:
                 * looping backwards will find the highest dimension left
                 * that isn't included in the dimension index list.
                 *
                 * This can also be thought of as the last item of the first index
                 * of the difference between the full list of indices and
                 * the dimension indices.
                 *
                 * We should avoid excessive object creation by only looping backwards.
                 */
                if(dimension[dimIdx--] != i) {
                    int ret = stride[i];
                    return ret;
                }
            }
        }




        int ret = stride[0];
        return ret;
    }

#ifdef __CUDACC__
    __device__ INLINEDEF void sweepShapeInfoBuffer(Nd4jLong *shapeInfoBuffer, Nd4jLong *targetBuffer) {
    // we read first element, to find out length of our shapeInfoBuffer
    int rank = shapeInfoBuffer[0];
    int len = shape::shapeInfoLength(rank);
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        targetBuffer[i] = shapeInfoBuffer[i];
}
#endif


    INLINEDEF _CUDA_HD Nd4jLong *shapeBufferOfNpy(cnpy::NpyArray arr) {
        return shape::shapeBufferOfNpy(arr.shape.size(),(unsigned int*) arr.shape.data(),arr.fortranOrder);
    }






//    INLINEDEF _CUDA_HD Nd4jLong *shapeBufferOfNpyBuffer(char *buffer) {
//        unsigned Nd4jLong *shape;
//        unsigned int ndims, wordSize;
//        bool fortranOrder;
//        cnpy::parseNpyHeaderStr(std::string(buffer),wordSize,shape,ndims,fortranOrder);
//        Nd4jLong * ret =  shape::shapeBufferOfNpy(ndims,shape,fortranOrder);
//        delete[] shape;
//        return ret;
//    }


    INLINEDEF _CUDA_HD Nd4jLong *shapeBufferOfNpy(int rank, unsigned int* shape,bool fortranOrder) {
        if(fortranOrder) {
            Nd4jLong *shapeBufferRet = shape::shapeBufferFortran(rank, nd4j::FLOAT32,(Nd4jLong *) shape);
            return shapeBufferRet;
        }
        else {
            Nd4jLong *newShape = new Nd4jLong[rank];
            for(int i = 0; i < rank; i++) {
                newShape[i] = shape[i];
            }

            Nd4jLong *shapeBufferRet = shape::shapeBuffer(rank, nd4j::FLOAT32, newShape);
            delete[] newShape;
            return shapeBufferRet;

        }
    }

    INLINEDEF _CUDA_HD bool strideDescendingCAscendingF(Nd4jLong *shapeBuffer) {
        int rank = shape::rank(shapeBuffer);
        Nd4jLong *strides = shape::stride(shapeBuffer);
        char order = shape::order(shapeBuffer);

        if (shape::isRowVector(shapeBuffer) && strides[0] == 1 && strides[1] == 1)
            return true;

        if (order == 'c') {
            for (int i = 1; i < rank; i++)
                if (strides[i-1] <= strides[i])
                    return false;
            return true;
        } else if (order == 'f') {
            for (int i = 1; i < rank; i++)
                if (strides[i-1] >= strides[i])
                    return false;
            return true;
        } else {
            printf("Unknown order for array!\n");
            return false;
        }
    }


    INLINEDEF _CUDA_H bool reshapeCF(const int oldRank, Nd4jLong* oldShape, const int newRank, Nd4jLong* newShapeOf, bool isFOrder, Nd4jLong* target) {
        int oldnd;
        Nd4jLong* olddims = shape::copyOf(oldRank, shape::shapeOf(oldShape));
        Nd4jLong* oldstrides = shape::copyOf(oldRank, shape::stride(oldShape));
        int np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        Nd4jLong* newStrides = new Nd4jLong[newRank];
        oldnd = 0;

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < oldRank; oi++) {
            if (shape::shapeOf(oldShape)[oi] != 1) {
                olddims[oldnd] = shape::shapeOf(oldShape)[oi];
                oldstrides[oldnd] = shape::stride(oldShape)[oi];
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newRank; ni++) {
            np *= newShapeOf[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
            /* different total sizes; no hope */
            delete[] olddims;
            delete[] oldstrides;
            delete[] newStrides;

            return false;
        }

        if (np == 0) {
            /* the current code does not handle 0-sized arrays, so give up */
            delete[] olddims;
            delete[] oldstrides;
            delete[] newStrides;

            return false;
        }

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;

        while (ni < newRank && oi < oldnd) {
            np = newShapeOf[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShapeOf[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        delete[] olddims;
                        delete[] oldstrides;
                        delete[] newStrides;

                        return false;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        delete[] olddims;
                        delete[] oldstrides;
                        delete[] newStrides;

                        return false;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShapeOf[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShapeOf[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        } else {
            last_stride = shape::elementWiseStride(oldShape);
        }
        if (isFOrder && ni >= 1) {
            last_stride *= newShapeOf[ni - 1];
        }
        for (nk = ni; nk < newRank; nk++) {
            newStrides[nk] = last_stride;
        }

        target[0] = newRank;
        int cnt = 1;
        for (int e = 0; e < newRank; e++)
            target[cnt++] = newShapeOf[e];

        for (int e = 0; e < newRank; e++)
            target[cnt++] = newStrides[e];

        target[shape::shapeInfoLength(newRank) - 3] = 0;
        target[shape::shapeInfoLength(newRank) - 2] = -1;
        target[shape::shapeInfoLength(newRank) - 1] = isFOrder ? 102 : 99;
        nd4j::ArrayOptions::setDataType(target, nd4j::ArrayOptions::dataType(oldShape));

        delete[] olddims;
        delete[] oldstrides;
        delete[] newStrides;

        return true;
    }

    INLINEDEF _CUDA_H bool canReshape(const int oldRank, Nd4jLong* oldShape, const int newRank, Nd4jLong* newShapeOf, bool isFOrder) {
        int oldnd;
        Nd4jLong* olddims = shape::copyOf(oldRank, shape::shapeOf(oldShape));
        Nd4jLong* oldstrides = shape::copyOf(oldRank, shape::stride(oldShape));
        int np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        auto newStrides = new Nd4jLong[newRank];
        oldnd = 0;

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < oldRank; oi++) {
            if (shape::shapeOf(oldShape)[oi] != 1) {
                olddims[oldnd] = shape::shapeOf(oldShape)[oi];
                oldstrides[oldnd] = shape::stride(oldShape)[oi];
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newRank; ni++) {
            np *= newShapeOf[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
            /* different total sizes; no hope */
            delete[] olddims;
            delete[] oldstrides;
            delete[] newStrides;

            return false;
        }

        if (np == 0) {
            /* the current code does not handle 0-sized arrays, so give up */
            delete[] olddims;
            delete[] oldstrides;
            delete[] newStrides;

            return false;
        }

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;

        while (ni < newRank && oi < oldnd) {
            np = newShapeOf[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShapeOf[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        delete[] olddims;
                        delete[] oldstrides;
                        delete[] newStrides;

                        return false;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        delete[] olddims;
                        delete[] oldstrides;
                        delete[] newStrides;

                        return false;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShapeOf[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShapeOf[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        delete[] olddims;
        delete[] oldstrides;
        delete[] newStrides;

        return true;
    }

    // this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too big number of dimensions)
    // also it sorts input array of dimensions, this operation is also necessary for creating TAD object
    INLINEDEF _CUDA_H void checkDimensions(const int rank, std::vector<int>& dimensions) {

        int dimSize = dimensions.size();
        if(dimSize == 0)
            throw std::runtime_error("shape::checkDimensions method: array of dimensions is empty!");
        // check presence of negative dimensions and if they are present transform them to positive ones -dim -> rank - |dim|
        for(auto& dim : dimensions)
            if(dim < 0)
                dim += rank;
        // sort input array of dimensions, this operation is also necessary for creating TAD object in external methods
        if (dimSize > 1) {
            std::sort(dimensions.begin(), dimensions.end());
            // remove duplicates if they are present
            dimensions.erase(std::unique(dimensions.begin(), dimensions.end()), dimensions.end());
        }
        // check whether number of dimensions is to big (>rank)
        dimSize = dimensions.size();
        if(dimSize > rank)
            throw std::runtime_error("shape::checkDimensions method: number of input dimensions is too big ( > rank of array)!");
        // check if min dimension is still negative and whether max dimension is bigger then rank-1
        if(dimensions[0] < 0 || dimensions.back() > (rank-1))
            throw std::runtime_error("shape::checkDimensions method: the negative dimension is still present in input array after transform or the too big dimension is present ( > rank of array) !");

    }



    // return absolute index of array min, min is sub-array of max, index to be returned is min's index and corresponds to maxIdx of max array
    INLINEDEF _CUDA_H Nd4jLong subArrayIndex(const Nd4jLong* maxShapeInfo, const Nd4jLong* minShapeInfo, const int maxIdx) {
        const auto rankMax = shape::rank(maxShapeInfo);
        const auto rankMin = shape::rank(minShapeInfo);

        Nd4jLong idxPerRank[MAX_RANK];
        ind2subC(rankMax, const_cast<Nd4jLong *>(maxShapeInfo)+1, const_cast<int&>(maxIdx), idxPerRank);

        Nd4jLong minIdx = 0;
        for(int i = 0; i < rankMin; ++i) {
            if(minShapeInfo[rankMin - i] == 1 || idxPerRank[rankMax - i - 1] == 0)
                continue;
            if(idxPerRank[rankMax - i - 1] >= minShapeInfo[rankMin - i])
                idxPerRank[rankMax - i - 1] %= minShapeInfo[rankMin - i];
            minIdx += idxPerRank[rankMax - i - 1] * stride(const_cast<Nd4jLong*>(minShapeInfo))[rankMin - i - 1];
        }

        return minIdx;
    }

    INLINEDEF _CUDA_HD void shapeOldScalar(nd4j::DataType dataType, Nd4jLong* const buffer, const char order) {

        buffer[0] = 2;
        buffer[1] = 1;
        buffer[2] = 1;
        buffer[3] = 1;
        buffer[4] = 1;
        buffer[6] = 1;
        buffer[7] = (int)order;

        nd4j::ArrayOptions::setDataType(buffer, dataType);
    }

    template <typename T1, typename T2>
    INLINEDEF _CUDA_H void convertT(T1 *from, T2 *to, Nd4jLong length) {
        for (Nd4jLong e = 0; e < length; e++)
                to[e] = (T2) from[e];
    };

}

#endif /* SHAPE_H_ */
        