//
// Created by raver119 on 20.11.17.
//

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
    int numIterations = 10000;
};


TEST_F(PlaygroundTests, LambdaTest_1) {
    NDArray<float> array('c', {128, 512});
    NDArrayFactory<float>::linspace(1, array);

    auto lambda = LAMBDA_F(_x) {
        return _x + 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < numIterations; e++) {
        array.applyLambda(lambda);
    }


    /*
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < numIterations; i++) {
        for (int e = 0; e < array.lengthOf(); e++) {
            array.buffer()[e] = array.buffer()[e] + 32.12f;
        }
    }
    */

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    nd4j_printf("Time %lld us\n", outerTime / numIterations);
}