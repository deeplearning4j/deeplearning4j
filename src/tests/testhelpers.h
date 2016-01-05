/*
 * testhelpers.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TESTHELPERS_H_
#define TESTHELPERS_H_
#include <CppUTest/TestHarness.h>
#include <templatemath.h>
int arrsEquals(int rank,int *comp1,int *comp2);

template <typename T>
int arrsEquals(int rank, T *comp1,T *comp2) {
	int ret = 1;
	for(int i = 0; i < rank; i++) {
		T eps = nd4j::math::nd4j_abs(comp1[i] - comp2[i]);
		ret = ret && (eps < 1e-3);
	}

	return ret;
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
int *shapeBuffer(int rank,int *shape) {
    int *stride = shape::calcStrides(shape,rank);
    shape::ShapeInformation * shapeInfo = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    shapeInfo->shape = shape;
    shapeInfo->stride = stride;
    shapeInfo->offset = 0;
    shapeInfo->rank = rank;
    int elementWiseStride = shape::computeElementWiseStride(rank,shape,stride,0);
    shapeInfo->elementWiseStride = elementWiseStride;
    int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
    free(shapeInfo);
    return shapeInfoBuffer;
}

void assertBufferProperties(int *shapeBuffer) {
    CHECK(shape::rank(shapeBuffer) >= 2);
    CHECK(shape::length(shapeBuffer) >= 1);
    CHECK(shape::elementWiseStride(shapeBuffer) >= 1);
}


#endif /* TESTHELPERS_H_ */
