//
// Created by agibsonccc on 1/5/16.
//

#ifndef NATIVEOPERATIONS_REDUCE3TESTS_H
#define NATIVEOPERATIONS_REDUCE3TESTS_H
#include <array.h>
#include "testhelpers.h"
#include <reduce3.h>
#include <shape.h>

TEST_GROUP(Reduce3)
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

TEST(Reduce3,EuclideanDistance) {
    functions::reduce3::Reduce3OpFactory<double> *opFactory6 = new functions::reduce3::Reduce3OpFactory<double>();
    functions::reduce3::Reduce3<double> *op = opFactory6->getOp("euclidean_strided");
    int vectorLength = 4;
    shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    int rank = 2;
    int shape[rank] = {1,vectorLength};
    int *stride = shape::calcStrides(shape,rank);
    vecShapeInfo->shape = shape;
    vecShapeInfo->stride = stride;
    vecShapeInfo ->offset = 0;
    vecShapeInfo->rank = rank;
    vecShapeInfo->elementWiseStride = 1;
    vecShapeInfo->order = 'c';
    int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
    assertBufferProperties(shapeInfo);
    double *result = (double *) malloc(sizeof(double));
    result[0] = 0.0;
    int *scalarShape = shape::createScalarShapeInfo();
    assertBufferProperties(scalarShape);

    double vec1[vectorLength] = {1,2,3,4};
    double vec2[vectorLength] = {5,6,7,8};
    double *extraParams = (double *) malloc(sizeof(double));
    extraParams[0] = 0.0;
    op->exec(vec1,shapeInfo,extraParams,vec2,shapeInfo,result,scalarShape);
    CHECK(8 == result[0]);


    free(result);

    free(scalarShape);
    free(vecShapeInfo);
    free(shapeInfo);
    delete (opFactory6);
    delete (op);
}

TEST(Reduce3,EuclideanDistanceDimension) {
    functions::reduce3::Reduce3OpFactory<double> *opFactory6 = new functions::reduce3::Reduce3OpFactory<double>();
    functions::reduce3::Reduce3<double> *op = opFactory6->getOp("euclidean_strided");
    int vectorLength = 4;
    shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    int rank = 2;
    int shape[rank] = {2,2};
    int *stride = shape::calcStrides(shape,rank);
    vecShapeInfo->shape = shape;
    vecShapeInfo->stride = stride;
    vecShapeInfo ->offset = 0;
    vecShapeInfo->rank = rank;
    vecShapeInfo->elementWiseStride = 1;
    vecShapeInfo->order = 'c';
    int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
    assertBufferProperties(shapeInfo);
    int resultLength = 2;
    double *result = (double *) malloc(resultLength * sizeof(double));
    for(int i = 0; i < resultLength; i++)
        result[i] = 0.0;
    //change to row vector
    int *scalarShape = shape::createScalarShapeInfo();
    shape::shapeOf(scalarShape)[1] = 2;

    assertBufferProperties(scalarShape);

    double vec1[vectorLength] = {1,2,3,4};
    double vec2[vectorLength] = {5,6,7,8};
    int dimensionLength = 1;
    int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
    dimension[0] = 1;

    double *extraParams = (double *) malloc(sizeof(double));
    extraParams[0] = 0.0;
    op->exec(vec1,shapeInfo,extraParams,vec2,shapeInfo,result,scalarShape,dimension,dimensionLength);

    double assertion[resultLength] = {5.656854249492381,5.6568542494923806};
    for(int i = 0; i < rank; i++)
        printf("Result %f\n",result[i]);
    CHECK(arrsEquals<double>(2,assertion,result));


    free(result);
    free(extraParams);

    free(scalarShape);
    free(vecShapeInfo);
    free(shapeInfo);
    delete (opFactory6);
    delete (op);
}
#endif //NATIVEOPERATIONS_REDUCE3TESTS_H
