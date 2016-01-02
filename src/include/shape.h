/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_


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
     int * calcStridesFortran(int *shape, int rank,int startNum);


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
     int * calcStrides(int *shape, int rank,int startNum);


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
     int computeElementWiseStride(int rank,int *shape,int *stride,int isFOrder);
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

void removeIndex(int *data, int *indexes, int dataLength, int indexesLength, int **out);


/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(shape::ShapeInformation *xInfo, int offset);

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

int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape, int tensorShapeLength, int *dimension,
		int dimensionLength);

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

int offset(int index, int rank, shape::ShapeInformation *info, int *dimension, int dimensionLength);


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

int tensorsAlongDimension(int rank, volatile int length, volatile int *shape, int *dimension,
		int dimensionLength);

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

int tensorsAlongDimension(TADPermuteInfo info, int *dimension, int dimensionLength);

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

int offset(int index, int *xShapeInfo, int *dimension, int dimensionLength, TADPermuteInfo info);

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
__device__ int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
__device__ int blockStartingIndex(int N);


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

int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced, int tadsForOriginal);

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

int reductionIndexForLinear(
		int i, int elementWiseStride, int numElementsPerTad, int tadNum, int originalTadNum);


/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
	__host__ __device__
#endif

int prod(int *data, int length);


}

#endif /* SHAPE_H_ */
