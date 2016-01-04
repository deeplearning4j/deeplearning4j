//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
#define NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H

#include <pairwise_transform.h>
#include <CppUTest/TestHarness.h>
#include <array.h>
#include <shape.h>
#include "testhelpers.h"



static functions::pairwise_transforms::PairWiseTransformOpFactory<double> *opFactory2 = 0;

TEST_GROUP(PairWiseTransform) {
    static int output_method(const char* output, ...)
    {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup()
    {
        opFactory2 = new  functions::pairwise_transforms::PairWiseTransformOpFactory<double>();

    }
    void teardown()
    {
        delete opFactory2;
    }
};


TEST(PairWiseTransform,Addition) {
    functions::pairwise_transforms::PairWiseTransform<double> *add = opFactory2->getOp("add_strided");
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    int *stride = shape::calcStrides(shape,rank);
    nd4j::array::NDArray<double> *data = nd4j::array::NDArrays<double>::createFrom(rank,shape,stride,0,0.0);
    int length = nd4j::array::NDArrays<double>::length(data);
    for(int i = 0; i < length; i++) {
        data->data->data[i] = i + 1;
        printf("Data[%d] is now %f\n",i,data->data->data[i]);
    }
    double *extraParams = (double *) malloc(sizeof(double));

    add->exec(data->data->data,1,data->data->data,1,data->data->data,1,extraParams,length);
    for(int i = 0; i < length; i++) {
        printf("Data[%d] is now %f\n",i,data->data->data[i]);
    }
    double comparison[4] = {2,4,6,8};
    CHECK(arrsEquals(rank,comparison,data->data->data));
    free(data);
    free(extraParams);
    free(shape);
    free(stride);
    delete add;

}


#endif //NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
