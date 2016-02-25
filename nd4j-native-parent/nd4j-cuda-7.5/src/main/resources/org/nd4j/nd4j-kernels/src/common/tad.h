/*
 * tad.h
 *
 *  Created on: Oct 21, 2015
 *      Author: agibsonccc
 */

#ifndef TAD_H_
#define TAD_H_

#define MAX_NUM_THREADS 1024
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mathutil.h"
#define MAX_DIMENSION  0x7fffffff

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
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
__device__ __host__ ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
	ShapeInformation *copy = (ShapeInformation *) malloc(sizeof(ShapeInformation));
	copy->shape = (int *) malloc(sizeof(int) * toCopy->rank);
	for(int i = 0; i < toCopy->rank; i++) {
		copy->shape[i] = toCopy->shape[i];
	}


	copy->stride = (int *) malloc(sizeof(int) * toCopy->rank);
	for(int i = 0; i < toCopy->rank; i++) {
		copy->stride[i] = toCopy->stride[i];
	}
	copy->order = toCopy->order;
	copy->rank = toCopy->rank;
	copy->offset = toCopy->offset;
	copy->elementWiseStride = toCopy->elementWiseStride;
	return copy;
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
__device__ __host__ void  removeIndex(int *data,int *indexes,int dataLength,int indexesLength,int **out) {
	int *ret = (int *) *out;
	int count = 0;
	int absLength = dataLength - indexesLength;
	for(int i = 0; i < dataLength && count < absLength; i++) {
		int contains = 0;
		for(int j = 0; j < indexesLength; j++) {
			if(i == indexes[j]) {
				contains = 1;
				break;
			}
		}

		if(!contains) {
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
__device__ int tadOffset(ShapeInformation *xInfo,int offset) {
	return offset + threadIdx.x  * xInfo->elementWiseStride;

}




/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
__device__ __host__ int*  ensureVectorShape(int *shape,int dimension) {
	int *ret = (int *) malloc(sizeof(int) * 2);
	if(dimension == 0) {
		ret[0] = 1;
		ret[1] = shape[0];
	}
	else {
		ret[0] = shape[0];
		ret[1] = 1;
	}
	return ret;
}


/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
__device__ __host__ int* range(int from,int to,int increment) {
	int diff = abs(from - to);
	int retLength = diff / increment;
	int *ret = diff / increment < 1 ? (int *) malloc(sizeof(int)) :  (int *) malloc(sizeof(int) * diff / increment);
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
 * Range between from and two with an
 * increment of 1
 */
__device__ __host__ int* range(int from,int to) {
	return range(from,to,1);
}

/**
 * Keep the given indexes
 * in the data
 */
__device__ __host__ int* keep(volatile int *data,int *index,int indexLength,int dataLength) {
	int *ret = (int *) malloc((indexLength) * sizeof(int));
	int count = 0;
	for(int i = 0; i < dataLength; i++) {
		int contains = 0;
		for(int j = 0; j < indexLength; j++) {
			if(i == index[j]) {
				contains = 1;
				break;
			}
		}

		if(contains)
			ret[count++] = data[i];
	}
	return ret;
}


/**
 * Generate a reverse
 * copy of the data
 */
__device__ __host__ int* reverseCopy(int *data,int length) {
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
 * Permutes the given dimensions
 */
__device__ __host__ int* doPermuteSwap(int length,int  *shape, int *rearrange) {
	int *ret = new int[length];
	for (int i = 0; i < length; i++) {
		ret[i] = shape[rearrange[i]];
	}
	return ret;
}


__device__ __host__ int checkArrangeArray(int *arr,int *shape,int arrLength,int shapeLength) {
	if(arrLength != shapeLength)
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
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
__device__ __host__ char getOrder(int length ,int *shape,int *stride,int elementStride) {
	int sd;
	int dim;
	int i;
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

	if(isFortran && cContiguous)
		return 'a';
	else if(isFortran && !cContiguous)
		return 'f';
	else if(!isFortran && !cContiguous)
		return 'c';
	else
		return 'c';

}

__device__ __host__ int* concat(int *arr1,int arr1Length,int *arr2,int arr2Length) {
	int *ret = (int *) malloc((arr1Length + arr2Length) * sizeof(int));
	memcpy(ret, arr1, arr1Length * sizeof(int));
	memcpy(ret + arr1Length, arr2, arr2Length * sizeof(int));
	return ret;
}



__device__ __host__ int* concat(int  numArrays,int numTotalElements,int **arr,int *lengths) {
	int *ret = (int *) malloc(numTotalElements * sizeof(int));
	int count = 0;
	for(int i = 0; i < numArrays; i++) {
		for(int j = 0; j < lengths[i]; j++) {
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
__device__ __host__ int lengthPerSlice(int rank,int *shape,int *dimension,int dimensionLength) {
	int *ret2 = (int *) malloc((abs(rank - dimensionLength)) * sizeof(int));
	removeIndex(shape,dimension,rank,dimensionLength,&ret2);
	int length = rank - dimensionLength;
	int ret = prod(ret2,length);
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
__device__ __host__ int sliceOffsetForTensor(int rank,int index, int *shape, int *tensorShape,int tensorShapeLength,int *dimension,int dimensionLength) {
	int tensorLength = prod(tensorShape,tensorShapeLength);
	int lengthPerSlice2 = lengthPerSlice(rank,shape,dimension,dimensionLength);
	if(lengthPerSlice2 <= 0)
		return 0;

	int offset = index * tensorLength / lengthPerSlice2;
	return offset;
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
__device__ __host__ int *copyOf(int length,int *toCopy) {
	int *ret = (int *) malloc(sizeof(int) * length);
	for(int i = 0; i < length; i++)
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
__device__ __host__ int * permutedStrides(int *toPermute,int shapeRank,int *rearrange) {
	int *strideCopy = copyOf(shapeRank,toPermute);
	checkArrangeArray(rearrange,strideCopy,shapeRank,shapeRank);
	int *newStride = doPermuteSwap(shapeRank,strideCopy,rearrange);
	free(strideCopy);
	return strideCopy;
}


/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
__device__ __host__ void permute(ShapeInformation **info,int *rearrange,int rank) {
	ShapeInformation *infoDeref = (ShapeInformation *) *info;
	checkArrangeArray(rearrange,infoDeref->shape,rank,rank);
	int *newShape = doPermuteSwap(rank,infoDeref->shape,rearrange);
	int *newStride = doPermuteSwap(rank,infoDeref->stride,rearrange);
	char order = getOrder(rank,infoDeref->shape,infoDeref ->stride,infoDeref->elementWiseStride);
	//free the old shape and strides
	free(infoDeref->shape);
	free(infoDeref->stride);
	infoDeref->shape = newShape;
	infoDeref->stride = newStride;
	infoDeref->order = order;

}

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
__device__ __host__ int *slice(int *shape) {
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
__device__ __host__ int shapeInfoLength(int rank) {
	return rank * 2 + 4;
}

/**
 * Returns the rank portion of
 * an information buffer
 */
__device__ __host__ int rank(int *buffer) {
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
__device__ __host__ ShapeInformation* infoFromBuffer(int *buffer) {
	ShapeInformation *info = (ShapeInformation *) malloc(sizeof(ShapeInformation));
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
 * Returns the shape portion of an information
 * buffer
 */
__device__ __host__ int * shape(int *buffer) {
	return buffer + 1;
}

/**
 * Returns the stride portion of an information
 * buffer
 */
__device__ __host__ int *stride(int *buffer) {
	return buffer + (1 + rank(buffer));
}

/**
 * Compute the length of the given shape
 */
__device__ __host__ int length(int *shapeInfo) {
	return prod(shape(shapeInfo),rank(shapeInfo));
}


/***
 * Returns the offset portion of an information buffer
 */
__device__ __host__ int offset(int *buffer) {
	int length = shapeInfoLength(rank(buffer));
	return buffer[length - 3];
}

/**
 * Returns the ordering
 * for this shape information buffer
 */
__device__ __host__ char order(int *buffer) {
	int length = buffer[0] * 2 + 4;
	return (char) buffer[length - 1];
}

/**
 * Returns the element wise stride for this information
 * buffer
 */
__device__ __host__ int elementWiseStride(int *buffer) {
	int length2 = shapeInfoLength(buffer[0]);
	return buffer[length2 - 2];
}

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
__device__ __host__ int isScalar(int *info) {
	if(rank(info) > 2)
		return 0;
	if(rank(info) == 1)
		return shape(info)[0] == 1;
	else if(rank(info) == 2) {
		return shape(info)[0] == 1 && shape(info)[1] == 1;
	}
	return 0;
}

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
__device__ __host__ int isScalar(volatile ShapeInformation *info) {
	if(info->rank > 2)
		return 0;
	if(info->rank == 1)
		return info->shape[0] == 1;
	else if(info->rank == 2) {
		return info->shape[0] == 1 && info->shape[1] == 1;
	}
	return 0;
}


/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(int *xInfo,int offset) {
	return offset + threadIdx.x  * elementWiseStride(xInfo);

}




/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
__device__ __host__ int isVector(int *shape,int rank) {
	if(rank > 2)
		return 0;
	else if(rank <= 2) {
		if(shape[0] == 1 || shape[1] == 1)
			return 1;
	}
	return 0;
}


/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
__device__ __host__ int offset(int index,int rank,ShapeInformation *info,int *dimension,int dimensionLength) {
	int  *tensorShape = keep(info->shape,dimension,dimensionLength,rank);
	if(dimensionLength == 1) {
		int *newTensorShape = ensureVectorShape(tensorShape,dimension[0]);
		free(tensorShape);
		tensorShape = newTensorShape;
	}

	//change the value
	ShapeInformation *copy = shapeCopy(info);
	info = copy;

	int  *reverseDimensions = reverseCopy(dimension,dimensionLength);
	int *rangeRet = range(0, rank);
	int *remove = (int *) malloc((rank - dimensionLength) * sizeof(int));
	removeIndex(rangeRet, dimension,rank,dimensionLength,&remove);

	int *zeroDimension = (int *) malloc(1 * sizeof(int));
	zeroDimension[0] = 0;

	int removeLength = rank - dimensionLength;
	int *newPermuteDims = concat(remove,removeLength,reverseDimensions,dimensionLength);

	//__device__ void permute(ShapeInformation *info,int *rearrange,int rank) {
	permute(&info,newPermuteDims,rank);

	int *permuted = info->shape;
	int *permutedStrides = info->stride;
	int tensorShapeLength = rank - removeLength;
	if(tensorShapeLength < 2)
		tensorShapeLength = 2;
	int sliceIdx = sliceOffsetForTensor(rank,index, permuted, tensorShape,tensorShapeLength,zeroDimension,1);

	//determine offset here

	int  *ret2 = slice(info->shape);
	int *ret2Stride = slice(info->stride);


	int ret2Length = prod(ret2,rank - 1);
	int ret2Rank = rank - 1;

	int retOffset = sliceIdx * permutedStrides[0];
	int tensorShapeProd = prod(tensorShape,tensorShapeLength);



	int length = prod(tensorShape,tensorShapeLength);
	int tensorLength = length;
	//__device__ int lengthPerSlice(int rank,int *shape,int *dimension) {
	int offset = index * tensorLength / lengthPerSlice(ret2Rank,ret2,zeroDimension,1);
	/**
	 * Need to do slice(offset) here
	 */
	if(sliceIdx == 0 && length == lengthPerSlice(ret2Rank,ret2,zeroDimension,1)) {
		/**
		 * NOTE STRIDE[1] HERE. WE DO THIS TO AVOID CREATING A NEW SLICE OBJECT.
		 */
		//account for shape[i] == 1
		int strideIndex = 1;
		for(int i = 0; i < info->rank; i++) {
			if(info->shape[i] == 1)
				strideIndex++;
		}

		if(strideIndex >= info->rank)
			strideIndex = info->rank - 1;

		retOffset = info->offset + offset  * info->stride[strideIndex];
	}

		//determine offset here
		//note here offset doesn't change, just the shape
		//of the tad
	else if(length == lengthPerSlice(ret2Rank,ret2,zeroDimension,1)) {
		offset -= ret2[0] * (offset / ret2[0]);
		//set offset here
		ret2 = slice(ret2);
		ret2Rank--;
		//account for shape[i] == 1
		int strideIndex = 1;
		for(int i = 0; i < info->rank; i++) {
			if(info->shape[i] == 1)
				strideIndex++;
		}

		if(strideIndex >= info->rank)
			strideIndex = info->rank - 1;

		retOffset += info->stride[strideIndex] * offset;
	}


	else {

		while(ret2Length > length) {
			sliceIdx = sliceOffsetForTensor(ret2Rank,index, ret2, tensorShape,tensorShapeLength,zeroDimension,1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			//set offset
			ret2 = slice(ret2);
			ret2Stride = slice(ret2Stride);
			ret2Rank--;
			//slice wise offsets are offset + i * majorStride()
			//dividing by the slice index will adjust the offset by a factor of sliceIndex
			ret2Length = prod(ret2,ret2Rank);

		}
	}

	retOffset = info->offset + sliceIdx;

	free(reverseDimensions);
	free(rangeRet);
	free(remove);
	free(copy);
	//free the new pointer
	if(rank <= 2) {
		free(tensorShape);
	}

	if(retOffset < 0)
		retOffset = 0;

	return  retOffset;
}


typedef struct  {
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
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
__device__ __host__ TADPermuteInfo tadInfo(int *xShapeInfo,int *dimension,int dimensionLength) {
	int *shapeOfX = shape(xShapeInfo);
	int xRank = rank(xShapeInfo);
	int  *tensorShape = keep(shapeOfX,dimension,dimensionLength,xRank);
	if(dimensionLength == 1) {
		int *newTensorShape = ensureVectorShape(tensorShape,dimension[0]);
		free(tensorShape);
		tensorShape = newTensorShape;
	}

	int removeLength = abs(xRank - dimensionLength);
	int tensorShapeLength = rank(xShapeInfo) - removeLength;
	if(tensorShapeLength < 2)
		tensorShapeLength = 2;


	int tensorShapeProd = prod(tensorShape,tensorShapeLength);


	int  *reverseDimensions = reverseCopy(dimension,dimensionLength);
	int *rangeRet = range(0, xRank);

	int *remove = (int *) malloc((removeLength) * sizeof(int));
	removeIndex(rangeRet, dimension,xRank,dimensionLength,&remove);

	int *zeroDimension = (int *) malloc(1 * sizeof(int));
	zeroDimension[0] = 0;

	int *newPermuteDims = concat(remove,removeLength,reverseDimensions,dimensionLength);


	int *permutedShape = doPermuteSwap(rank(xShapeInfo),shape(xShapeInfo),newPermuteDims);
	int *permutedStrides = doPermuteSwap(rank(xShapeInfo),stride(xShapeInfo),newPermuteDims);

	TADPermuteInfo info = {
			tensorShape,
			xRank,
			reverseDimensions,
			rangeRet,
			removeLength,
			remove,
			zeroDimension,
			newPermuteDims,
			permutedShape,
			permutedStrides,
			tensorShapeLength,
			tensorShapeProd
	};

	return info;
}

/**
 * Frees the permute information
 * @param info the info to free
 */
__host__ __device__ void freePermuteInfo(TADPermuteInfo info) {
	free(info.tensorShape);
	free(info.reverseDimensions);
	free(info.rangeRet);
	free(info.remove);
	free(info.zeroDimension);
	free(info.newPermuteDims);
	free(info.permutedShape);
	free(info.permutedStrides);

}


__host__ __device__ void printIntArray(int *arr,int length) {
	for(int i = 0; i < length; i++) {
		printf("arr[%d] is %d \n",i,arr[i]);
	}
}


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
__device__ __host__ int tensorsAlongDimension(volatile int rank,volatile int length,volatile int *shape,int *dimension,int dimensionLength) {
	int *tensorShape = keep(shape,dimension,dimensionLength,rank);
	int ret = length / prod(tensorShape,dimensionLength);
	free(tensorShape);
	return ret;
}

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
__device__ __host__ int tensorsAlongDimension(int *shapeInfo,int *dimension,int dimensionLength) {
	int *tensorShape = keep(shape(shapeInfo),dimension,dimensionLength,rank(shapeInfo));
	int ret = length(shapeInfo) / prod(tensorShape,dimensionLength);
	free(tensorShape);
	return ret;
}

/**
 *
 */
__device__ __host__ int tensorsAlongDimension(TADPermuteInfo info,int *dimension,int dimensionLength) {
	int length = prod(info.permutedShape,info.xRank);
	return length / prod(info.tensorShape, info.tensorShapeLength);
}

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
__device__ __host__ int offset(int index,int *xShapeInfo,int *dimension,int dimensionLength,TADPermuteInfo info) {
	int sliceIdx = sliceOffsetForTensor(rank(xShapeInfo),index, info.permutedShape, info.tensorShape,info.tensorShapeLength,info.zeroDimension,1);

	//determine offset here

	int  *ret2 = slice(info.permutedShape);
	int *ret2Stride = slice(info.permutedStrides);
	int ret2Length = prod(ret2,rank(xShapeInfo) - 1);
	int ret2Rank = info.xRank - 1;

	int retOffset = sliceIdx * info.permutedStrides[0];
	int tensorShapeProd = info.tensorShapeProd;

	int tensorShapeRoughlyEquals = dimensionLength == 1 && abs(info.tensorShapeLength - dimensionLength) <= 1;
	if((tensorShapeProd == ret2Length && tensorShapeRoughlyEquals == 1) || dimensionLength == info.tensorShapeLength) {
		return retOffset;
	}


	int length = info.tensorShapeProd;
	int tensorLength = length;
	int sliceOffset = index * tensorLength / lengthPerSlice(ret2Rank,ret2,info.zeroDimension,1);
	/**
	 * Need to do slice(offset) here
	 */
	int lengthPerSlice2 =  lengthPerSlice(ret2Rank,ret2,info.zeroDimension,1);

	if(sliceIdx == 0 && length == lengthPerSlice2) {
		ret2 = slice(ret2);
		ret2Stride = slice(ret2Stride);
		ret2Rank--;
		ret2Length = prod(ret2,ret2Rank);
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += ( sliceOffset * ret2Length * newStride);

		if(retOffset < 0)
			retOffset = 0;

		return  retOffset;
	}

		//determine offset here
		//note here offset doesn't change, just the shape
		//of the tad
	else if(length == lengthPerSlice2) {
		sliceOffset -= ret2[0] * (sliceOffset / ret2[0]);
		//set offset here
		ret2 = slice(ret2);
		ret2Stride = slice(ret2Stride);
		ret2Rank--;
		//accumulate from the slice
		int newStride = ret2Stride[ret2Rank - 1];
		retOffset += (lengthPerSlice2 * newStride * sliceOffset);

		if(retOffset < 0)
			retOffset = 0;

		return  retOffset;
	}


	else {
		ret2Length = prod(ret2,ret2Rank);
		//start at zero incrementing whenever we hit a slice > 0
		while(ret2Length > length && ret2Rank > 0) {
			sliceIdx = sliceOffsetForTensor(ret2Rank,index, ret2, info.tensorShape,info.tensorShapeLength,info.zeroDimension,1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			if(sliceIdx > 0) {
				if(ret2Rank > 1) {
					retOffset += sliceIdx * ret2Stride[0];
				}
				else {
					retOffset += sliceIdx;
				}
			}
			//set offset
			ret2 = slice(ret2);
			ret2Stride = slice(ret2Stride);
			//bump the offset wrt the slice idx when its not just truncating output

			ret2Rank--;
			ret2Length = prod(ret2,ret2Rank);
		}

		return retOffset;
	}


}

__device__ __host__ int tadForBlockIndex(int blockSize,int blockIdx,int i) {
	int ret = blockIdx + i * blockSize;
	return ret;
}

/**
 * Computes the number of tads per block
 *
 */
__device__ __host__ int tadsPerBlock(int blockSize,int tads) {
	return  ceil(tads / (double) blockSize);
}

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
__device__ __host__ int * toShapeBuffer(ShapeInformation *info) {
	int *ret = new int[shapeInfoLength(info->rank)];
	int count = 1;
	ret[0] = info->rank;
	for(int i = 0; i < info->rank; i++) {
		ret[count++] = info->shape[i];
	}
	for(int i = 0; i < info->rank; i++) {
		ret[count++] = info->stride[i];
	}

	ret[count++] = info->offset;
	ret[count++] = info->elementWiseStride;
	ret[count++] = info->order;


	return ret;
}





#endif /* TAD_H_ */
