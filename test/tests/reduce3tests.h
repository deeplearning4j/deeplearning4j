//
// Created by agibsonccc on 1/5/16.
//

#ifndef NATIVEOPERATIONS_REDUCE3TESTS_H
#define NATIVEOPERATIONS_REDUCE3TESTS_H
#include <array.h>
#include "testhelpers.h"
#include <reduce3.h>
#include <shape.h>

TEST_GROUP(Reduce3) {

	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {

	}
	void teardown() {
	}
};

TEST(Reduce3,CosineSimilarity) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(2);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int*) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = vectorLength;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);
	double *result = (double *) malloc(sizeof(double));
	result[0] = 0.0;
	int *scalarShape = shape::createScalarShapeInfo();
	assertBufferProperties(scalarShape);

	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);

	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = i + 1;
	}

	int extraParamsLength = 3;
	double *extraParams = (double *) malloc(extraParamsLength * sizeof(double));
	extraParams[0] = 0.0;
	extraParams[1] = 5.4772255750516612;
	extraParams[2] = 5.4772255750516612;
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result,
			scalarShape);
	CHECK(1.0 == result[0]);

	free(result);
	free(vec1);
	free(vec2);
	free(scalarShape);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}

TEST(Reduce3,EuclideanDistance) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(1);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = vectorLength;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);
	double *result = (double *) malloc(sizeof(double));
	result[0] = 0.0;
	int *scalarShape = shape::createScalarShapeInfo();
	assertBufferProperties(scalarShape);
	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);

	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = vec1[i] + 4;
	}

	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result,
			scalarShape);
	CHECK(8 == result[0]);

	free(shape);
	free(result);
	free(vec1);
	free(vec2);
	free(scalarShape);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}

TEST(Reduce3,EuclideanDistanceDimension) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(1);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);
	int resultLength = 2;
	double *result = (double *) malloc(resultLength * sizeof(double));
	for (int i = 0; i < resultLength; i++)
		result[i] = 0.0;
	//change to row vector
	int *scalarShape = shape::createScalarShapeInfo();
	shape::shapeOf(scalarShape)[1] = 2;

	assertBufferProperties(scalarShape);
	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);

	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = vec1[i] + 4;
	}

	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
	dimension[0] = 1;

	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result, scalarShape,
			dimension, dimensionLength);

	double *assertion = (double *) malloc(sizeof(double) * resultLength);
	assertion[0] = 5.656854249492381;
	assertion[1] = 5.6568542494923806;
	for (int i = 0; i < rank; i++)
		printf("Result %f\n", result[i]);
	CHECK(arrsEquals<double>(2, assertion, result));

	free(vec1);
	free(vec2);
	free(shape);
	free(result);
	free(extraParams);
	free(assertion);
	free(scalarShape);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}
#endif //NATIVEOPERATIONS_REDUCE3TESTS_H
