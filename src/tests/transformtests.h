//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_TRANSFORMTESTS_H
#define NATIVEOPERATIONS_TRANSFORMTESTS_H

#include <transform.h>
#include <CppUTest/TestHarness.h>
#include <array.h>
#include <shape.h>
#include "testhelpers.h"



static functions::transform::TransformOpFactory<double> *opFactory = 0;

TEST_GROUP(Transform) {
    static int output_method(const char* output, ...)
    {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup()
    {
        if(!opFactory) {
            opFactory = new  functions::transform::TransformOpFactory<double>();
        }


    }
    void teardown()
    {
        free(opFactory);
    }
};

TEST(Transform,Log) {
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    int *stride = shape::calcStrides(shape,rank);
    nd4j::array::NDArray<double> *data = nd4j::array::NDArrays<double>::createFrom(rank,shape,stride,0,0.0);
    int length = nd4j::array::NDArrays<double>::length(data);
    for(int i = 0; i < length; i++) {
        data->data[i] = i + 1;
    }

    functions::transform::Transform<double> *log = opFactory->getOp("log_strided");
    log->exec(data->data->data,1,data->data->data,1,0,1);
    double comparison[4] = {0.,0.69314718,1.09861229,1.38629436};
    CHECK(arrsEquals(rank,comparison,data->data->data));
    nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
    free(shape);
    free(stride);






}



#endif //NATIVEOPERATIONS_TRANSFORMTESTS_H
