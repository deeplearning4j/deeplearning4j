/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include <templatemath.h>
#include <cstring>
#include <stdlib.h>
namespace shape {
const int MAX_DIMENSION = 0x7fffffff;
const int MAX_NUM_THREADS = 1024;

/**
 * Shape information approximating
 * the information on an ndarray
 */
typedef struct {
	int *shape;
	int *stride;
	char order;
	int rank;
	int offset;
	int elementWiseStride;
} ShapeInformation;

/**
 * Indexing information
 * for bounds checking
 */
typedef struct {
	int numElementsPerThread;
	int blockStartingIndex;
	int startingThreadIndex;
	int endingThreadIndex;

} CurrentIndexing;

/**
 * TADPermuteInfo is for intermediate information
 * needed for computing tensor along dimension.
 *
 *
 */
typedef struct {
	int *tensorShape;
	int xRank;
	int *reverseDimensions;
	int *rangeRet;
	int removeLength;
	int *remove;
	int *zeroDimension;
	int *newPermuteDims;
	int *permutedShape;
	int *permutedStrides;
	int tensorShapeLength;
	int tensorShapeProd;
} TADPermuteInfo;


#ifdef __CUDACC__
__inline__ __host__ __device__
#endif

int tadIndexForLinear(int linearIndex, int tadLength);
/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int *shapeBuffer(int rank, int *shape);
/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStridesFortran(int *shape, int rank);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStrides(int *shape, int rank);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStridesFortran(int *shape, int rank, int startNum);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStrides(int *shape, int rank, int startNum);

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
#ifdef __CUDACC__
__host__ __device__
#endif

ShapeInformation *shapeCopy(ShapeInformation *toCopy);

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
#ifdef __CUDACC__
__host__ __device__
#endif
int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder);

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
#ifdef __CUDACC__
__host__ __device__
#endif
int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder,
		int *dimension, int dimensionLength);

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *doPermuteSwap(int length, int *shape, int *rearrange);

/**
 * In place permute swap
 * @param length
 * @param shape
 * @param rearrange
 */
#ifdef __CUDACC__
__host__ __device__
#endif

void doPermuteSwap(int length, int **shape, int *rearrange);

/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

char getOrder(int length, int *shape, int *stride, int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int checkArrangeArray(int *arr, int *shape, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
#ifdef __CUDACC__
__host__ __device__
#endif

void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int isVector(int *shape, int rank);

/**
 * Returns the shape portion of an information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *shapeOf(int *buffer);

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *copyOf(int length, int *toCopy);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *permutedStrides(int *toPermute, int shapeRank, int *rearrange);

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *slice(int *shape);

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int shapeInfoLength(int rank);

/**
 * Returns the rank portion of
 * an information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int rank(int *buffer);

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
#ifdef __CUDACC__
__host__ __device__
#endif

ShapeInformation *infoFromBuffer(int *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *stride(int *buffer);

/**
 * Compute the length of the given shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int length(int *shapeInfo);

/***
 * Returns the offset portion of an information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int *buffer);

/**
 * Returns the ordering
 * for this shape information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

char order(int *buffer);

/**
 * Returns the element wise stride for this information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int elementWiseStride(int *buffer);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int isScalar(int *info);

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int isScalar(volatile ShapeInformation *info);

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
#ifdef __CUDACC__
__host__ __device__
#endif

void removeIndex(int *data, int *indexes, int dataLength, int indexesLength,
		int **out);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
__device__
#endif
int tadOffset(shape::ShapeInformation *xInfo, int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *ensureVectorShape(int *shape, int dimension);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *ensureVectorShape(int *shape);

#ifdef __CUDACC__
__host__ __device__
#endif
int * createScalarShapeInfo();

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *range(int from, int to);

/**
 * Keep the given indexes
 * in the data
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *keep(volatile int *data, int *index, int indexLength, int dataLength);

/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int *reverseCopy(int *data, int length);

/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length);

/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *concat(int numArrays, int numTotalElements, int **arr, int *lengths);

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
#ifdef __CUDACC__
__host__ __device__
#endif

int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape,
		int tensorShapeLength, int *dimension, int dimensionLength);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
__device__
#endif
int tadOffset(int *xInfo, int offset);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int index, int rank, shape::ShapeInformation *info, int *dimension,
		int dimensionLength);

/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
#ifdef __CUDACC__
__host__ __device__
#endif

TADPermuteInfo tadInfo(int *xShapeInfo, int *dimension, int dimensionLength);

/**
 * Frees the permute information
 * @param info the info to free
 */
#ifdef __CUDACC__
__host__ __device__
#endif

void freePermuteInfo(TADPermuteInfo info);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(int rank, volatile int length, volatile int *shape,
		int *dimension, int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength);

/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(TADPermuteInfo info);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int index, int *xShapeInfo, int dimensionLength,
		TADPermuteInfo info);

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tadForBlockIndex(int blockSize, int blockIdx, int i);

/**
 * Computes the number of tads per block
 *
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tadsPerBlock(int blockSize, int tads);

#ifdef __CUDACC__
__host__ __device__
#endif

int * tadShapeInfo(int index, int *xShapeInfo, int *dimension,
		int dimensionLength);

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *toShapeBuffer(ShapeInformation *info);

/**

 * Returns the number of elements per thread
 */
#ifdef __CUDACC__
__device__
#endif
int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
#ifdef __CUDACC__
__device__
#endif
int blockStartingIndex(int N);

/**
 * Returns the thread starting index
 */
#ifdef __CUDACC__
__device__
#endif
int threadStartingIndex(int N, int stride, int offset);

/**
 * Returns the thread ending index
 */
#ifdef __CUDACC__
__device__
#endif
int threadEndingIndex(int N, int stride, int offset);

/**
 * Returns indexing information
 * for the current kernel invocation
 */
#ifdef __CUDACC__
__device__
#endif
CurrentIndexing *currentIndex(int N, int offset, int stride);

/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int tadIndex(int i, int elementWiseStride, int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
		int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
		int tadNum, int originalTadNum);

/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int prod(int *data, int length);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStridesFortran(int *shape, int rank, int startNum) {
	if (isVector(shape, rank)) {
		int *ret = (int *) malloc(sizeof(int) * 2);
		for (int i = 0; i < 2; i++)
			ret[i] = 1;
		return ret;

	}

	int dimensions = rank;
	int *stride = (int *) malloc(sizeof(int) * dimensions);
	int st = startNum;
	for (int j = 0; j < rank; j++) {
		stride[j] = st;
		st *= shape[j];
	}

	return stride;
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStrides(int *shape, int rank, int startNum) {
	if (shape::isVector(shape, rank)) {
		int *ret = (int *) malloc(sizeof(int) * 2);
		for (int i = 0; i < 2; i++)
			ret[i] = 1;
		return ret;

	}

	int dimensions = rank;
	int *stride = (int *) malloc(sizeof(int) * dimensions);

	int st = startNum;
	for (int j = dimensions - 1; j >= 0; j--) {
		stride[j] = st;
		st *= shape[j];
	}

	return stride;
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStridesFortran(int *shape, int rank) {
	return calcStridesFortran(shape, rank, 1);
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif
int * calcStrides(int *shape, int rank) {
	return calcStrides(shape, rank, 1);
}

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
#ifdef __CUDACC__
__host__ __device__
#endif

ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
	ShapeInformation *copy = (ShapeInformation *) malloc(
			sizeof(ShapeInformation));
	copy->shape = (int *) malloc(sizeof(int) * toCopy->rank);
	for (int i = 0; i < toCopy->rank; i++) {
		copy->shape[i] = toCopy->shape[i];
	}

	copy->stride = (int *) malloc(sizeof(int) * toCopy->rank);
	for (int i = 0; i < toCopy->rank; i++) {
		copy->stride[i] = toCopy->stride[i];
	}
	copy->order = toCopy->order;
	copy->rank = toCopy->rank;
	copy->offset = toCopy->offset;
	copy->elementWiseStride = toCopy->elementWiseStride;
	return copy;
}

#ifdef __CUDACC__
__host__ __device__
#endif
int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder) {
	int oldnd;
	int *olddims = shape::copyOf(rank, shape);
	int *oldstrides = shape::copyOf(rank, stride);
	int np, op, last_stride;
	int oi, oj, ok, ni, nj, nk;
	int *newStrides = (int *) malloc(sizeof(int) * rank);
	oldnd = 0;
	//set the shape to be 1 x length
	int newShapeRank = 2;
	int *newShape = (int *) malloc(sizeof(int) * newShapeRank);
	newShape[0] = 1;
	newShape[1] = shape::prod(shape, rank);

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
		return -1;
	}

	if (np == 0) {
		/* the current code does not handle 0-sized arrays, so give up */
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
					return -1;
				}
			} else {
				/* C order */
				if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
					/* not contiguous enough */
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
	free(newStrides);
	free(newShape);
	free(oldstrides);
	free(olddims);
	return ret;
}

#ifdef __CUDACC__
__host__ __device__
#endif
int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder,
		int *dimension, int dimensionLength) {
	if(dimensionLength == 1) {
		return stride[dimension[0]];
	}
	return -1;

}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
int *shapeBuffer(int rank, int *shape) {
	int *stride = shape::calcStrides(shape, rank);
	shape::ShapeInformation * shapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	shapeInfo->shape = shape;
	shapeInfo->stride = stride;
	shapeInfo->offset = 0;
	shapeInfo->rank = rank;
	int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride,
			0);
	shapeInfo->elementWiseStride = elementWiseStride;
	int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
	free(shapeInfo);
	return shapeInfoBuffer;
}

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *doPermuteSwap(int length, int *shape, int *rearrange) {
	int *ret = (int *) malloc(sizeof(int) * length);
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
#ifdef __CUDACC__
__host__ __device__
#endif

void doPermuteSwap(int length, int **shape, int *rearrange) {
	int *shapeDeref = *shape;
	for (int i = 0; i < length; i++) {
		int x = shapeDeref[i];
		int j = i;
		while (1) {
			int k = rearrange[j];
			rearrange[j] = j;
			if (k == i)
				break;
			shapeDeref[j] = shapeDeref[k];
			j = k;

		}

		shapeDeref[j] = x;
	}

}

/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

char getOrder(int length, int *shape, int *stride, int elementStride) {
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
#ifdef __CUDACC__
__host__ __device__
#endif

int checkArrangeArray(int *arr, int *shape, int arrLength, int shapeLength) {
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

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
#ifdef __CUDACC__
__host__ __device__
#endif

void permute(ShapeInformation **info, int *rearrange, int rank) {
	ShapeInformation *infoDeref = *info;
	checkArrangeArray(rearrange, infoDeref->shape, rank, rank);
	shape::doPermuteSwap(rank, &infoDeref->shape, rearrange);
	shape::doPermuteSwap(rank, &infoDeref->stride, rearrange);
	char order = getOrder(rank, infoDeref->shape, infoDeref->stride,
			infoDeref->elementWiseStride);
	infoDeref->order = order;

}

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int isVector(int *shape, int rank) {
	if (rank > 2)
		return 0;
	else if (rank <= 2) {
		if (shape[0] == 1 || shape[1] == 1)
			return 1;
	}
	return 0;
}

/**
 * Returns the shape portion of an information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *shapeOf(int *buffer) {
	return buffer + 1;
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *copyOf(int length, int *toCopy) {
	int *ret = (int *) malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
		ret[i] = toCopy[i];
	return ret;
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
#ifdef __CUDACC__
__host__ __device__
#endif
int *permutedStrides(int *toPermute, int shapeRank, int *rearrange) {
	int *strideCopy = copyOf(shapeRank, toPermute);
	checkArrangeArray(rearrange, strideCopy, shapeRank, shapeRank);
	int *newStride = doPermuteSwap(shapeRank, strideCopy, rearrange);
	free(strideCopy);
	return newStride;
}

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *slice(int *shape) {
	return shape + 1;
}

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int shapeInfoLength(int rank) {
	return rank * 2 + 4;
}

/**
 * Returns the rank portion of
 * an information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int rank(int *buffer) {
	return buffer[0];
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
#ifdef __CUDACC__
__host__ __device__
#endif

ShapeInformation *infoFromBuffer(int *buffer) {
	ShapeInformation *info = (ShapeInformation *) malloc(
			sizeof(ShapeInformation));
	int length = shapeInfoLength(rank(buffer));
	int rank = buffer[0];

	//start after rank
	info->shape = buffer + 1;
	info->stride = buffer + (1 + rank);
	info->rank = rank;
	info->offset = buffer[length - 3];
	info->elementWiseStride = buffer[length - 2];
	int *stride = buffer + 1 + rank;
	info->stride = stride;
	info->order = (char) buffer[length - 1];
	return info;
}

/**
 * Returns the stride portion of an information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *stride(int *buffer) {
	return buffer + (1 + rank(buffer));
}

/**
 * Compute the length of the given shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int length(int *shapeInfo) {
	return shape::prod(shape::shapeOf(shapeInfo), shape::rank(shapeInfo));
}

/***
 * Returns the offset portion of an information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int *buffer) {
	int length = shape::shapeInfoLength(shape::rank(buffer));
	return buffer[length - 3];
}

/**
 * Returns the ordering
 * for this shape information buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

char order(int *buffer) {
	int length = buffer[0] * 2 + 4;
	return (char) buffer[length - 1];
}

/**
 * Returns the element wise stride for this information
 * buffer
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int elementWiseStride(int *buffer) {
	int length2 = shapeInfoLength(buffer[0]);
	return buffer[length2 - 2];
}

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int isScalar(int *info) {
	if (shape::rank(info) > 2)
		return 0;
	if (shape::rank(info) == 1)
		return shape::shapeOf(info)[0] == 1;
	else if (rank(info) == 2) {
		return shape::shapeOf(info)[0] == 1 && shape::shapeOf(info)[1] == 1;
	}
	return 0;
}

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int isScalar(volatile ShapeInformation *info) {
	if (info->rank > 2)
		return 0;
	if (info->rank == 1)
		return info->shape[0] == 1;
	else if (info->rank == 2) {
		return info->shape[0] == 1 && info->shape[1] == 1;
	}
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
#ifdef __CUDACC__
__host__ __device__
#endif

void removeIndex(int *data, int *indexes, int dataLength, int indexesLength,
		int **out) {
	int *ret = *out;
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
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
__device__ int tadOffset(ShapeInformation *xInfo, int offset) {
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
#ifdef __CUDACC__
__host__ __device__
#endif

int *ensureVectorShape(int *shape, int dimension) {
	int *ret = (int *) malloc(sizeof(int) * 2);
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
#ifdef __CUDACC__
__host__ __device__
#endif

int *ensureVectorShape(int *shape) {
	return ensureVectorShape(shape, 0);
}

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *range(int from, int to, int increment) {
	int diff = nd4j::math::nd4j_abs<int>(from - to);
	int retLength = diff / increment;
	int *ret =
			diff / increment < 1 ?
					(int *) malloc(sizeof(int)) :
					(int *) malloc(sizeof(int) * diff / increment);
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
#ifdef __CUDACC__
__host__ __device__
#endif

int *range(int from, int to) {
	return range(from, to, 1);
}

/**
 * Keep the given indexes in the data
 * @param data
 * @param index
 * @param indexLength
 * @param dataLength
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *keep(volatile int *data, int *index, int indexLength, int dataLength) {
	int *ret = (int *) malloc((indexLength) * sizeof(int));
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
#ifdef __CUDACC__
__host__ __device__
#endif

int *reverseCopy(int *data, int length) {
	if (length < 1)
		return data;

	int *copy = (int *) malloc(length * sizeof(int));
	for (int i = 0; i <= length / 2; i++) {
		int temp = data[i];
		copy[i] = data[length - i - 1];
		copy[length - i - 1] = temp;
	}
	return copy;
}

/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length) {
	int *ret = (int *) malloc((arr1Length + arr2Length) * sizeof(int));
	std::memcpy(ret, arr1, arr1Length * sizeof(int));
	std::memcpy(ret + arr1Length, arr2, arr2Length * sizeof(int));
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
#ifdef __CUDACC__
__host__ __device__
#endif

int *concat(int numArrays, int numTotalElements, int **arr, int *lengths) {
	int *ret = (int *) malloc(numTotalElements * sizeof(int));
	int count = 0;
	for (int i = 0; i < numArrays; i++) {
		for (int j = 0; j < lengths[i]; j++) {
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
#ifdef __CUDACC__
__host__ __device__
#endif

int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength) {
	int absSelta = nd4j::math::nd4j_abs<int>(rank - dimensionLength);
	int *ret2 = (int *) malloc(absSelta * sizeof(int));
	removeIndex(shape, dimension, rank, dimensionLength, &ret2);
	int length = rank - dimensionLength;
	int ret = prod(ret2, length);
	free(ret2);
	return ret;
}

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape,
		int tensorShapeLength, int *dimension, int dimensionLength) {
	int tensorLength = prod(tensorShape, tensorShapeLength);
	int lengthPerSlice2 = lengthPerSlice(rank, shape, dimension,
			dimensionLength);
	if (lengthPerSlice2 <= 0)
		return 0;

	int offset = index * tensorLength / lengthPerSlice2;
	return offset;
}
#ifdef __CUDACC__

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(int *xInfo, int offset) {
	return offset + threadIdx.x * elementWiseStride(xInfo);

}
#endif

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int index, int rank, ShapeInformation *info, int *dimension,
		int dimensionLength) {
	int *tensorShape = keep(info->shape, dimension, dimensionLength, rank);
	if (dimensionLength == 1) {
		int *newTensorShape = ensureVectorShape(tensorShape, dimension[0]);
		free(tensorShape);
		tensorShape = newTensorShape;
	}

	//change the value
	ShapeInformation *copy = shapeCopy(info);
	info = copy;

	int *reverseDimensions = reverseCopy(dimension, dimensionLength);
	int *rangeRet = range(0, rank);
	int *remove = (int *) malloc((rank - dimensionLength) * sizeof(int));
	removeIndex(rangeRet, dimension, rank, dimensionLength, &remove);

	int *zeroDimension = (int *) malloc(1 * sizeof(int));
	zeroDimension[0] = 0;

	int removeLength = rank - dimensionLength;
	int *newPermuteDims = concat(remove, removeLength, reverseDimensions,
			dimensionLength);

	//__device__ void permute(ShapeInformation *info,int *rearrange,int rank) {
	permute(&info, newPermuteDims, rank);

	int *permuted = info->shape;
	int *permutedStrides = info->stride;
	int tensorShapeLength = rank - removeLength;
	if (tensorShapeLength < 2)
		tensorShapeLength = 2;
	int sliceIdx = sliceOffsetForTensor(rank, index, permuted, tensorShape,
			tensorShapeLength, zeroDimension, 1);

	//determine offset here

	int *ret2 = slice(info->shape);
	int *ret2Stride = slice(info->stride);

	int ret2Length = prod(ret2, rank - 1);
	int ret2Rank = rank - 1;

	int retOffset = sliceIdx * permutedStrides[0];

	int length = prod(tensorShape, tensorShapeLength);
	int tensorLength = length;
	//__device__ int lengthPerSlice(int rank,int *shape,int *dimension) {
	int offset = index * tensorLength
			/ lengthPerSlice(ret2Rank, ret2, zeroDimension, 1);
	/**
	 * Need to do slice(offset) here
	 */
	if (sliceIdx == 0
			&& length == lengthPerSlice(ret2Rank, ret2, zeroDimension, 1)) {
		/**
		 * NOTE STRIDE[1] HERE. WE DO THIS TO AVOID CREATING A NEW SLICE OBJECT.
		 */
		//account for shape[i] == 1
		int strideIndex = 1;
		for (int i = 0; i < info->rank; i++) {
			if (info->shape[i] == 1)
				strideIndex++;
		}

		if (strideIndex >= info->rank)
			strideIndex = info->rank - 1;

		retOffset = info->offset + offset * info->stride[strideIndex];
	}

	//determine offset here
	//note here offset doesn't change, just the shape
	//of the tad
	else if (length == lengthPerSlice(ret2Rank, ret2, zeroDimension, 1)) {
		offset -= ret2[0] * (offset / ret2[0]);
		//set offset here
		ret2 = slice(ret2);
		ret2Rank--;
		//account for shape[i] == 1
		int strideIndex = 1;
		for (int i = 0; i < info->rank; i++) {
			if (info->shape[i] == 1)
				strideIndex++;
		}

		if (strideIndex >= info->rank)
			strideIndex = info->rank - 1;

		retOffset += info->stride[strideIndex] * offset;
	}

	else {

		while (ret2Length > length) {
			sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2, tensorShape,
					tensorShapeLength, zeroDimension, 1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			//set offset
			ret2 = slice(ret2);
			ret2Stride = slice(ret2Stride);
			ret2Rank--;
			//slice wise offsets are offset + i * majorStride()
			//dividing by the slice index will adjust the offset by a factor of sliceIndex
			ret2Length = prod(ret2, ret2Rank);

		}
	}

	retOffset = info->offset + sliceIdx;

	free(reverseDimensions);
	free(rangeRet);
	free(remove);
	free(copy);
	//free the new pointer
	if (rank <= 2) {
		free(tensorShape);
	}

	if (retOffset < 0)
		retOffset = 0;

	return retOffset;
}

/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
#ifdef __CUDACC__
__host__ __device__
#endif

TADPermuteInfo tadInfo(int *xShapeInfo, int *dimension, int dimensionLength) {
	int *shapeOfX = shape::shapeOf(xShapeInfo);
	int xRank = shape::rank(xShapeInfo);
	int *tensorShape = shape::keep(shapeOfX, dimension, dimensionLength, xRank);
	if (dimensionLength == 1) {
		int *newTensorShape = shape::ensureVectorShape(tensorShape,
				dimension[0]);
		free(tensorShape);
		tensorShape = newTensorShape;
	}

	int removeLength = nd4j::math::nd4j_abs<int>(xRank - dimensionLength);
	int tensorShapeLength = shape::rank(xShapeInfo) - removeLength;
	if (tensorShapeLength < 2)
		tensorShapeLength = 2;

	int tensorShapeProd = shape::prod(tensorShape, tensorShapeLength);

	int *reverseDimensions = shape::reverseCopy(dimension, dimensionLength);
	int *rangeRet = shape::range(0, xRank);

	int *remove = (int *) malloc((removeLength) * sizeof(int));
	shape::removeIndex(rangeRet, dimension, xRank, dimensionLength, &remove);

	int *zeroDimension = (int *) malloc(1 * sizeof(int));
	zeroDimension[0] = 0;

	int *newPermuteDims = shape::concat(remove, removeLength, reverseDimensions,
			dimensionLength);
	int *permutedShape = shape::copyOf(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo));
	int *permutedStrides = shape::copyOf(shape::rank(xShapeInfo),shape::stride(xShapeInfo));
	shape::doPermuteSwap(shape::rank(xShapeInfo),&permutedShape,newPermuteDims);
	shape::doPermuteSwap(shape::rank(xShapeInfo),&permutedStrides,newPermuteDims);
	TADPermuteInfo info = { tensorShape, xRank, reverseDimensions, rangeRet,
			removeLength, remove, zeroDimension, newPermuteDims, permutedShape,
			permutedStrides, tensorShapeLength, tensorShapeProd };

	return info;
}

/**
 * Frees the permute information
 * @param info the info to free
 */
#ifdef __CUDACC__
__host__ __device__
#endif

void freePermuteInfo(TADPermuteInfo info) {
	free(info.tensorShape);
	free(info.reverseDimensions);
	free(info.rangeRet);
	free(info.remove);
	free(info.zeroDimension);
	free(info.newPermuteDims);
	free(info.permutedShape);
	free(info.permutedStrides);

}

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(volatile int rank, volatile int length,
		volatile int *shape, int *dimension, int dimensionLength) {
	int *tensorShape = shape::keep(shape, dimension, dimensionLength, rank);
	int ret = length / shape::prod(tensorShape, dimensionLength);
	free(tensorShape);
	return ret;
}

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength) {
	int *keepShape = shape::shapeOf(shapeInfo);
	int *tensorShape = shape::keep(keepShape, dimension, dimensionLength,
			rank(shapeInfo));
	int ret = shape::length(shapeInfo)
	/ shape::prod(tensorShape, dimensionLength);
	free(tensorShape);
	return ret;
}

/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tensorsAlongDimension(TADPermuteInfo info) {
	int length = shape::prod(info.permutedShape, info.xRank);
	return length / shape::prod(info.tensorShape, info.tensorShapeLength);
}

/**
 *
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int * tadShapeInfo(int index, int *xShapeInfo, int *dimension,
		int dimensionLength) {
	TADPermuteInfo tadInfo = shape::tadInfo(xShapeInfo, dimension,
			dimensionLength);
	int sliceIdx = shape::sliceOffsetForTensor(rank(xShapeInfo), index,
			tadInfo.permutedShape, tadInfo.tensorShape,
			tadInfo.tensorShapeLength, tadInfo.zeroDimension, 1);

	//no more dimensions
	if (tadInfo.xRank <= 2) {
		shape::freePermuteInfo(tadInfo);
		return xShapeInfo;
	}

	//determine offset here

	int *ret2 = shape::slice(tadInfo.permutedShape);
	int *ret2Stride = shape::slice(tadInfo.permutedStrides);
	int ret2Length = shape::prod(ret2, shape::rank(xShapeInfo) - 1);
	int ret2Rank = tadInfo.xRank - 1;

	int retOffset = sliceIdx * tadInfo.permutedStrides[0];
	int tensorShapeProd = tadInfo.tensorShapeProd;
	int delta = nd4j::math::nd4j_abs<int>(
			tadInfo.tensorShapeLength - dimensionLength);
	int tensorShapeRoughlyEquals = dimensionLength == 1 && delta <= 1;
	if ((tensorShapeProd == ret2Length && tensorShapeRoughlyEquals == 1)
			|| dimensionLength == tadInfo.tensorShapeLength) {
		ShapeInformation *info = (ShapeInformation *) malloc(
				sizeof(ShapeInformation));
		//row vector
		if (ret2Rank == 1) {
			ret2Rank++;
			ret2 = shape::ensureVectorShape(ret2);
			ret2Stride = shape::ensureVectorShape(ret2Stride);
		}
		info->shape = ret2;
		info->stride = ret2Stride;
		info->offset = retOffset;
		info->rank = ret2Rank;
		int *shapeInfoRet = shape::toShapeBuffer(info);
		free(info);
		shape::freePermuteInfo(tadInfo);
		return shapeInfoRet;
	}

	int length = tadInfo.tensorShapeProd;
	int tensorLength = length;
	int sliceOffset = index * tensorLength
			/ shape::lengthPerSlice(ret2Rank, ret2, tadInfo.zeroDimension, 1);
	/**
	 * Need to do slice(offset) here
	 */
	int lengthPerSlice2 = shape::lengthPerSlice(ret2Rank, ret2,
			tadInfo.zeroDimension, 1);

	if (sliceIdx == 0 && length == lengthPerSlice2) {
		ret2 = slice(ret2);
		ret2Stride = shape::slice(ret2Stride);
		ret2Rank--;
		ret2Length = shape::prod(ret2, ret2Rank);
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += (sliceOffset * ret2Length * newStride);

		if (retOffset < 0)
			retOffset = 0;
		//row vector
		if (ret2Rank == 1) {
			ret2Rank++;
			ret2 = shape::ensureVectorShape(ret2);
			ret2Stride = shape::ensureVectorShape(ret2Stride);
		}
		ShapeInformation *info = (ShapeInformation *) malloc(
				sizeof(ShapeInformation));
		info->shape = ret2;
		info->stride = ret2Stride;
		info->offset = retOffset;
		info->rank = ret2Rank;
		int *shapeInfoRet = shape::toShapeBuffer(info);
		free(info);
		shape::freePermuteInfo(tadInfo);
		return shapeInfoRet;
	}

	//determine offset here
	//note here offset doesn't change, just the shape
	//of the tad
	else if (length == lengthPerSlice2) {
		sliceOffset -= ret2[0] * (sliceOffset / ret2[0]);
		//set offset here
		ret2 = slice(ret2);
		ret2Stride = slice(ret2Stride);
		ret2Rank--;
		//accumulate from the slice
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += (lengthPerSlice2 * newStride * sliceOffset);

		if (retOffset < 0)
			retOffset = 0;

		ShapeInformation *info = (ShapeInformation *) malloc(
				sizeof(ShapeInformation));
		//row vector
		if (ret2Rank == 1) {
			ret2Rank++;
			ret2 = ensureVectorShape(ret2);
			ret2Stride = ensureVectorShape(ret2Stride);
		}
		info->shape = ret2;
		info->stride = ret2Stride;
		info->offset = retOffset;
		info->rank = ret2Rank;
		int *shapeInfoRet = shape::toShapeBuffer(info);
		free(info);
		shape::freePermuteInfo(tadInfo);
		return shapeInfoRet;
	}

	else {
		ret2Length = prod(ret2, ret2Rank);
		//start at zero incrementing whenever we hit a slice > 0
		while (ret2Length > length && ret2Rank > 0) {
			sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2,
					tadInfo.tensorShape, tadInfo.tensorShapeLength,
					tadInfo.zeroDimension, 1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			if (sliceIdx > 0) {
				if (ret2Rank > 1) {
					retOffset += sliceIdx * ret2Stride[0];
				} else {
					retOffset += sliceIdx;
				}
			}
			//set offset
			ret2 = shape::slice(ret2);
			ret2Stride = shape::slice(ret2Stride);
			//bump the offset wrt the slice idx when its not just truncating output

			ret2Rank--;
			ret2Length = shape::prod(ret2, ret2Rank);
		}

		ShapeInformation *info = (ShapeInformation *) malloc(
				sizeof(ShapeInformation));
		//row vector
		if (ret2Rank == 1) {
			ret2Rank++;
			ret2 = ensureVectorShape(ret2);
			ret2Stride = ensureVectorShape(ret2Stride);
		}
		info->shape = ret2;
		info->stride = ret2Stride;
		info->offset = retOffset;
		info->rank = ret2Rank;
		int *shapeInfoRet = shape::toShapeBuffer(info);
		free(info);
		shape::freePermuteInfo(tadInfo);
		return shapeInfoRet;
	}

}

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int offset(int index, int *xShapeInfo, int dimensionLength,
		TADPermuteInfo info) {
	int sliceIdx = sliceOffsetForTensor(rank(xShapeInfo), index,
			info.permutedShape, info.tensorShape, info.tensorShapeLength,
			info.zeroDimension, 1);

	//determine offset here

	int *ret2 = slice(info.permutedShape);
	int *ret2Stride = slice(info.permutedStrides);
	int ret2Length = prod(ret2, rank(xShapeInfo) - 1);
	int ret2Rank = info.xRank - 1;

	int retOffset = sliceIdx * info.permutedStrides[0];
	int tensorShapeProd = info.tensorShapeProd;
	int val = nd4j::math::nd4j_abs<int>(
			info.tensorShapeLength - dimensionLength) <= 1;
	int tensorShapeRoughlyEquals = dimensionLength == 1 && val;
	if ((tensorShapeProd == ret2Length && tensorShapeRoughlyEquals == 1)
			|| dimensionLength == info.tensorShapeLength) {
		return retOffset;
	}

	int length = info.tensorShapeProd;
	int tensorLength = length;
	int sliceOffset = index * tensorLength
			/ lengthPerSlice(ret2Rank, ret2, info.zeroDimension, 1);
	/**
	 * Need to do slice(offset) here
	 */
	int lengthPerSlice2 = lengthPerSlice(ret2Rank, ret2, info.zeroDimension, 1);

	if (sliceIdx == 0 && length == lengthPerSlice2) {
		ret2 = slice(ret2);
		ret2Stride = slice(ret2Stride);
		ret2Rank--;
		ret2Length = prod(ret2, ret2Rank);
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += (sliceOffset * ret2Length * newStride);

		if (retOffset < 0)
			retOffset = 0;

		return retOffset;
	}

	//determine offset here
	//note here offset doesn't change, just the shape
	//of the tad
	else if (length == lengthPerSlice2) {
		sliceOffset -= ret2[0] * (sliceOffset / ret2[0]);
		//set offset here
		ret2 = slice(ret2);
		ret2Stride = slice(ret2Stride);
		ret2Rank--;
		//accumulate from the slice
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += (lengthPerSlice2 * newStride * sliceOffset);

		if (retOffset < 0)
			retOffset = 0;

		return retOffset;
	}

	else {
		ret2Length = prod(ret2, ret2Rank);
		//start at zero incrementing whenever we hit a slice > 0
		while (ret2Length > length && ret2Rank > 0) {
			sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2,
					info.tensorShape, info.tensorShapeLength,
					info.zeroDimension, 1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			if (sliceIdx > 0) {
				if (ret2Rank > 1) {
					retOffset += sliceIdx * ret2Stride[0];
				} else {
					retOffset += sliceIdx;
				}
			}
			//set offset
			ret2 = slice(ret2);
			ret2Stride = slice(ret2Stride);
			//bump the offset wrt the slice idx when its not just truncating output

			ret2Rank--;
			ret2Length = prod(ret2, ret2Rank);
		}

		return retOffset;
	}

}

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
#ifdef __CUDACC__
__device__ __host__
#endif
int tadForBlockIndex(int blockSize, int blockIdx, int i) {
	int ret = blockIdx + i * blockSize;
	return ret;
}

/**
 * Computes the number of tads per block
 *
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tadsPerBlock(int blockSize, int tads) {
	return nd4j::math::nd4j_ceil<double>(tads / (double) blockSize);
}

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int *toShapeBuffer(ShapeInformation *info) {
	int *ret = (int *) malloc(sizeof(int) * shapeInfoLength(info->rank));
	int count = 1;
	ret[0] = info->rank;
	for (int i = 0; i < info->rank; i++) {
		ret[count++] = info->shape[i];
	}
	for (int i = 0; i < info->rank; i++) {
		ret[count++] = info->stride[i];
	}

	ret[count++] = info->offset;
	ret[count++] = info->elementWiseStride;
	ret[count++] = info->order;

	return ret;
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
#ifdef __CUDACC__
__host__ __device__
#endif

int tadIndex(int i, int elementWiseStride, int numElementsPerTad) {
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
#ifdef __CUDACC__
__host__ __device__
#endif

int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
		int tadsForOriginal) {
	if (tadIndexForOriginal == 0)
		return 0;
	return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
}

/**
 * Tad index for linear
 * @param linearIndex
 * @param tadLength
 * @return
 */
#ifdef __CUDACC__
__inline__ __host__ __device__
#endif

int tadIndexForLinear(int linearIndex, int tadLength) {
	return linearIndex % tadLength;
}

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal) {
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
#ifdef __CUDACC__
__host__ __device__
#endif
int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
		int tadNum, int originalTadNum) {
	int tad = tadIndex(i, elementWiseStride, numElementsPerTad);
	return reductionIndexForTad(tad, tadNum, originalTadNum);
}

#ifdef __CUDACC__
__host__ __device__
#endif
int * createScalarShapeInfo() {
	int *shape = (int *) malloc(sizeof(int) * 2);
	shape[0] = 1;
	shape[1] = 1;
	int *stride = (int *) malloc(sizeof(int) * 2);
	stride[0] = 1;
	stride[1] = 1;
	ShapeInformation *shapeInformation2 = (ShapeInformation *) malloc(
			sizeof(ShapeInformation));
	shapeInformation2->rank = 2;
	shapeInformation2->offset = 0;
	shapeInformation2->stride = stride;
	shapeInformation2->shape = shape;
	shapeInformation2->elementWiseStride = 1;
	int *ret = shape::toShapeBuffer(shapeInformation2);
	free(shapeInformation2);
	return ret;
}

/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
__host__ __device__
#endif

int prod(int *data, int length) {
	int prod = 1;
	for (int i = 0; i < length; i++) {
		prod *= data[i];
	}

	return prod;
}

}

#endif /* SHAPE_H_ */
