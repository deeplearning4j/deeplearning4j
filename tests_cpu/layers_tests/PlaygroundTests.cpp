//
// Created by raver119 on 20.11.17.
//

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>

using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
    int numIterations = 10;
    int poolSize = 10;

    PlaygroundTests() {
        printf("\n");
        fflush(stdout);
    }
};



TEST_F(PlaygroundTests, LambdaTest_1) {
    NDArray<float> array('c', {8192, 1024});
    NDArrayFactory<float>::linspace(1, array);

    auto lambda = LAMBDA_F(_x) {
        return _x + 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < numIterations; e++) {
        array.applyLambda(lambda);
    }


    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    nd4j_printf("Lambda 1 time %lld us\n", outerTime / numIterations);
}




TEST_F(PlaygroundTests, LambdaTest_2) {
    NDArray<float> array('c', {8192, 1024});
    NDArray<float> row('c', {1, 1024});
    NDArrayFactory<float>::linspace(1, array);

    auto lambda = LAMBDA_F(_x) {
        return _x + 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < numIterations; e++) {
        array.template applyBroadcast<simdOps::Add<float>>({1}, &row);
    }


    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    nd4j_printf("Broadcast time %lld us\n", outerTime / numIterations);
}


TEST_F(PlaygroundTests, NoCacheTest_1) {
    std::vector<NDArray<float> *> pool(poolSize);
    NDArray<float> source('c', {8192, 1024});
    for (int e = 0; e < pool.size(); e++)
        pool[e] = source.dup();

    auto lambda = LAMBDA_F(_x) {
        return _x * 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    for (int e = 0; e < numIterations; e++) {
        auto v = pool[poolSize - 1 - (cnt++)];
        v->applyLambda(lambda);

        if (cnt == poolSize)
            cnt = 0;
    }

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    nd4j_printf("Non-cached time %lld us\n", outerTime / numIterations);

    for (auto v: pool)
        delete v;
}


TEST_F(PlaygroundTests, NoCacheTest_2) {
    std::vector<NDArray<float> *> pool1(poolSize);
    std::vector<NDArray<float> *> pool2(poolSize);
    NDArray<float> source('c', {8192, 1024});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }

    auto lambda = LAMBDA_FF(_x, _y) {
        return _x * 32.12f + _y;
    };

    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    for (int e = 0; e < numIterations; e++) {
        auto v1 = pool1[poolSize - 1 - cnt];
        auto v2 = pool2[cnt++];
        v1->applyPairwiseLambda(v2, lambda);

        if (cnt == poolSize)
            cnt = 0;
    }

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    nd4j_printf("Non-cached PWT time %lld us\n", outerTime / numIterations);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ReductionTest_1) {
    std::vector<NDArray<float> *> pool1(poolSize);
    std::vector<NDArray<float> *> pool2(poolSize);
    NDArray<float> source('c', {1, 100});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }


    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    for (int e = 0; e < 100000; e++) {
        auto v = pool1[poolSize - 1 - cnt];
        auto r = v->sumNumber();

        if (cnt == poolSize)
            cnt = 0;
    }
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
    auto outerTimeMs = std::chrono::duration_cast<std::chrono::milliseconds> (timeEnd - timeStart).count();

    nd4j_printf("Non-cached reduction time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000, outerTimeMs);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ScalarTest_1) {
    std::vector<NDArray<float> *> pool1(poolSize);
    std::vector<NDArray<float> *> pool2(poolSize);
    NDArray<float> source('c', {1, 100});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }


    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    float *buff = source.buffer();
    for (int e = 0; e < 100000; e++) {
        //auto v = pool1[poolSize - 1 - cnt];
        //v->template applyScalar<simdOps::Add<float>>(2.0f);
        source.template applyScalar<simdOps::Add<float>>(2.0f);
        //functions::scalar::ScalarTransform<float>::template transformEx<simdOps::Add<float>>(source.buffer(), 1, source.buffer(), 1, 2.0f, nullptr, source.lengthOf());
        //functions::scalar::ScalarTransform<float>::template transform<simdOps::Add<float>>(buff, 1, buff, 1, 2.0f, nullptr, 100);

        cnt++;
        if (cnt == poolSize)
            cnt = 0;
    }
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
    auto outerTimeMs = std::chrono::duration_cast<std::chrono::milliseconds> (timeEnd - timeStart).count();

    nd4j_printf("Cached scalar time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000L, outerTimeMs);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ScalarTest_2) {
    std::vector<NDArray<float> *> pool1(poolSize);
    std::vector<NDArray<float> *> pool2(poolSize);
    NDArray<float> source('c', {1, 100});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }


    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    float * array = source.buffer();
    for (int e = 0; e < 100000; e++) {

#pragma omp simd
        for (int i = 0; i < source.lengthOf(); i++) {
            array[i] = simdOps::Add<float>::op(array[i], 2.0f);
        }

        cnt++;
        if (cnt == poolSize)
            cnt = 0;
    }
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
    auto outerTimeMs = std::chrono::duration_cast<std::chrono::milliseconds> (timeEnd - timeStart).count();

    nd4j_printf("Cached manual scalar time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000, outerTimeMs);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}

TEST_F(PlaygroundTests, Test_Profile_1) {
    GraphProfile prof;

    prof.setBuildTime(70);
    prof.setExecutionTime(130);

    prof.startEvent("omega");
    prof.spotEvent("alpha");
    prof.spotEvent("beta");
    prof.spotEvent("gamma");
    prof.recordEvent("omega");

    auto nodeA = prof.nodeById(1, "MatMul");
    auto nodeB = prof.nodeById(2, "Sum");
    auto nodeC = prof.nodeById(3, "Conv2D");

    nodeA->setObjectsSize(512);
    nodeA->setTemporarySize(65536);
    nodeA->setActivationsSize(512387);
    nodeA->setPreparationTime(127);
    nodeA->setExecutionTime(6539);


    nodeB->setObjectsSize(0);
    nodeB->setTemporarySize(0);
    nodeB->setActivationsSize(512387);
    nodeB->setPreparationTime(132);
    nodeB->setExecutionTime(2047);


    nodeC->setObjectsSize(1536);
    nodeC->setTemporarySize(2355674);
    nodeC->setActivationsSize(1022092);
    nodeC->setPreparationTime(129);
    nodeC->setExecutionTime(12983);

    prof.printOut();
}


TEST_F(PlaygroundTests, Test_Profile_2) {
    Environment::getInstance()->setProfiling(true);
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");

    auto profile = GraphProfilingHelper<float>::profile(graph, 1000);
    profile->printOut();

    delete graph;
    delete profile;
}