//
// Created by agibsonccc on 1/4/16.
//

#ifndef NATIVEOPERATIONS_BROADCASTSTESTS_H
#define NATIVEOPERATIONS_BROADCASTSTESTS_H
#include <broadcasting.h>
#include <CppUTest/TestHarness.h>
#include <array.h>
#include <shape.h>
#include "testhelpers.h"

static functions::pairwise_transforms::PairWiseTransformOpFactory<double> *opFactory3 = 0;

TEST_GROUP(BroadCasting) {
    static int output_method(const char* output, ...)
    {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup()
    {
        opFactory3 = new  functions::pairwise_transforms::PairWiseTransformOpFactory<double>();

    }
    void teardown()
    {
        delete opFactory3;
    }
};


TEST(BroadCasting,Addition) {
    functions::pairwise_transforms::PairWiseTransform<double> *add = opFactory2->getOp("add_strided");
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    int *stride = shape::calcStrides(shape,rank);
    nd4j::array::NDArray<double> *data = nd4j::array::NDArrays<double>::createFrom(rank,shape,stride,0,0.0);
    int length = nd4j::array::NDArrays<double>::length(data);

    double *extraParams = (double *) malloc(sizeof(double));

    add->exec(data->data->data,1,data->data->data,1,data->data->data,1,extraParams,length);
    double comparison[4] = {2,4,6,8};
    CHECK(arrsEquals(rank,comparison,data->data->data));
    free(data);
    free(extraParams);
    free(shape);
    free(stride);
    delete add;

}




#endif //NATIVEOPERATIONS_BROADCASTSTESTS_H
