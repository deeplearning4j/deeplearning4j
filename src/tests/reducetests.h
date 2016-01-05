/*
 * reducetests.h
 *
 *  Created on: Dec 31, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCETESTS_H_
#define REDUCETESTS_H_
#include <array.h>
#include "testhelpers.h"
#include <reduce.h>

TEST_GROUP(Reduce)
{

    static int output_method(const char* output, ...)
    {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup()
    {

    }
    void teardown()
    {
    }
};

TEST(Reduce, Sum) {
    functions::reduce::ReduceOpFactory<double> *opFactory5 = new functions::reduce::ReduceOpFactory<double>();
    functions::reduce::ReduceFunction<double> *sum = opFactory5->create(std::string("sum"));
    CHECK(sum != NULL);
    int length = 4;
    double *data = (double *) malloc(sizeof(double) * length);
    for(int i = 0; i < length; i++) {
        data[i] = i + 1;
    }
    int *resultShapeInfo = shape::createScalarShapeInfo();

    shape::ShapeInformation *shapeInfo = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 1;
    shape[1] = length;
    int *stride = shape::calcStrides(shape,rank);
    shapeInfo->shape = shape;
    shapeInfo->stride = stride;
    shapeInfo->offset = 0;
    shapeInfo->elementWiseStride = 1;

    int *shapeBuffer = shape::toShapeBuffer(shapeInfo);
    double *extraParams = (double *) malloc(sizeof(double));
    extraParams[0] = 0.0;

    double *result = (double *) malloc(sizeof(double));
    result[0] = 0.0;
    sum->exec(data,shapeBuffer,extraParams,result,resultShapeInfo);
    double comp = result[0];
    CHECK(10.0 ==comp);
    free(extraParams);
    free(shapeBuffer);
    free(shapeInfo);
    delete sum;
    free(data);
    delete opFactory5;


}

TEST(Reduce,DimensionSum) {
    functions::reduce::ReduceOpFactory<double> *opFactory5 = new functions::reduce::ReduceOpFactory<double>();
    functions::reduce::ReduceFunction<double> *sum = opFactory5->create(std::string("sum"));
    CHECK(sum != NULL);
    int length = 4;
    double *data = (double *) malloc(sizeof(double) * length);
    for(int i = 0; i < length; i++) {
        data[i] = i + 1;
    }
    int *resultShapeInfo = shape::createScalarShapeInfo();

    shape::ShapeInformation *shapeInfo = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    int *stride = shape::calcStrides(shape,rank);
    shapeInfo->shape = shape;
    shapeInfo->stride = stride;
    shapeInfo->offset = 0;
    shapeInfo->elementWiseStride = 1;

    int *shapeBuffer = shape::toShapeBuffer(shapeInfo);
    double *extraParams = (double *) malloc(sizeof(double));
    extraParams[0] = 0.0;

    int resultLength = 2;
    double *result = (double *) malloc(resultLength * sizeof(double));
    for(int i = 0; i < resultLength; i++)
        result[i] = 0.0;
    int dimensionLength = 1;
    int *dimension = (int *) malloc(dimensionLength* sizeof(int));
    dimension[0] = 1;

    sum->exec(data,shapeBuffer,extraParams,result,resultShapeInfo,dimension,dimensionLength);
    double comp[resultLength] {3.0,7.0};
    CHECK(arrsEquals(2,comp,result));
    free(extraParams);
    free(dimension);
    free(shapeBuffer);
    free(shapeInfo);
    delete sum;
    free(data);
    delete opFactory5;
}




#endif /* REDUCETESTS_H_ */
