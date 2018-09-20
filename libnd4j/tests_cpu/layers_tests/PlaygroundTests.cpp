/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 20.11.17.
//

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>

using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
    int numIterations = 3;
    int poolSize = 10;

    PlaygroundTests() {
        printf("\n");
        fflush(stdout);
    }
};



TEST_F(PlaygroundTests, LambdaTest_1) {
    auto array = NDArrayFactory::_create<float>('c', {8192, 1024});
    array.linspace(1);

    auto lambda = LAMBDA_F(_x) {
        return _x + 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < numIterations; e++) {
        array.applyLambda<float>(lambda);
    }


    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Lambda 1 time %lld us\n", outerTime / numIterations);
}




TEST_F(PlaygroundTests, LambdaTest_2) {
    auto array = NDArrayFactory::_create<float>('c', {8192, 1024});
    auto row = NDArrayFactory::_create<float>('c', {1, 1024});
    array.linspace(1);

    auto lambda = LAMBDA_F(_x) {
        return _x + 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < numIterations; e++) {
        array.applyBroadcast(broadcast::Add, {1}, &row);
    }


    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Broadcast time %lld us\n", outerTime / numIterations);
}


TEST_F(PlaygroundTests, NoCacheTest_1) {
    std::vector<NDArray*> pool(poolSize);
    auto source = NDArrayFactory::_create<float>('c', {8192, 1024});
    for (int e = 0; e < pool.size(); e++)
        pool[e] = source.dup();

    auto lambda = LAMBDA_F(_x) {
        return _x * 32.12f;
    };

    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    for (int e = 0; e < numIterations; e++) {
        auto v = pool[poolSize - 1 - (cnt++)];
        v->applyLambda<float>(lambda);

        if (cnt == poolSize)
            cnt = 0;
    }

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Non-cached time %lld us\n", outerTime / numIterations);

    for (auto v: pool)
        delete v;
}


TEST_F(PlaygroundTests, NoCacheTest_2) {
    std::vector<NDArray*> pool1(poolSize);
    std::vector<NDArray*> pool2(poolSize);
    auto source = NDArrayFactory::_create<float>('c', {8192, 1024});
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
        v1->applyPairwiseLambda<float>(v2, lambda);

        if (cnt == poolSize)
            cnt = 0;
    }

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Non-cached PWT time %lld us\n", outerTime / numIterations);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ReductionTest_1) {
    std::vector<NDArray*> pool1(poolSize);
    std::vector<NDArray*> pool2(poolSize);
    auto source = NDArrayFactory::_create<float>('c', {1, 100});
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

    // nd4j_printf("Non-cached reduction time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000, outerTimeMs);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ScalarTest_1) {
    std::vector<NDArray*> pool1(poolSize);
    std::vector<NDArray*> pool2(poolSize);
    auto source = NDArrayFactory::_create<float>('c', {1, 100});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }


    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    float *buff = reinterpret_cast<float*>(source.buffer());
    for (int e = 0; e < 100000; e++) {
        //auto v = pool1[poolSize - 1 - cnt];
        //v->template applyScalar<simdOps::Add<float>>(2.0f);
        source.applyScalar(scalar::Add,2.0f);
        //functions::scalar::ScalarTransform<float>::template transformEx<simdOps::Add<float>>(source.buffer(), 1, source.buffer(), 1, 2.0f, nullptr, source.lengthOf());
        //functions::scalar::ScalarTransform<float>::template transform<simdOps::Add<float>>(buff, 1, buff, 1, 2.0f, nullptr, 100);

        cnt++;
        if (cnt == poolSize)
            cnt = 0;
    }
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
    auto outerTimeMs = std::chrono::duration_cast<std::chrono::milliseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Cached scalar time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000L, outerTimeMs);

    for (auto v: pool1)
        delete v;

    for (auto v: pool2)
        delete v;
}


TEST_F(PlaygroundTests, ScalarTest_2) {
    std::vector<NDArray*> pool1(poolSize);
    std::vector<NDArray*> pool2(poolSize);
    auto source = NDArrayFactory::_create<float>('c', {1, 100});
    for (int e = 0; e < pool1.size(); e++) {
        pool1[e] = source.dup();
        pool2[e] = source.dup();
    }


    auto timeStart = std::chrono::system_clock::now();
    int cnt = 0;
    float * array = reinterpret_cast<float*>(source.buffer());
    for (int e = 0; e < 100000; e++) {

#pragma omp simd
        for (int i = 0; i < source.lengthOf(); i++) {
            array[i] = simdOps::Add<float, float>::op(array[i], 2.0f);
        }

        cnt++;
        if (cnt == poolSize)
            cnt = 0;
    }
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
    auto outerTimeMs = std::chrono::duration_cast<std::chrono::milliseconds> (timeEnd - timeStart).count();

    // nd4j_printf("Cached manual scalar time avg: %lld ns; Total time: %lld ms;\n", outerTime / 100000, outerTimeMs);

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

    // prof.printOut();
}


TEST_F(PlaygroundTests, Test_Profile_2) {
    Environment::getInstance()->setProfiling(true);
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");

    auto profile = GraphProfilingHelper::profile(graph, 2);
    // profile->printOut();

    delete graph;
    delete profile;
}

TEST_F(PlaygroundTests, Test_Im2Col_1) {
    
    int bS=16, iH=224,iW=224,  iC=3,oC=3,  kH=11,kW=11,  sH=4,sW=4,  pH=2,pW=2,  dH=1,dW=1;    
    int        oH=55, oW=55;
    int iterations = 1;

    auto input = NDArrayFactory::_create<float>('c', {bS, iC, iH, iW});
    auto output = NDArrayFactory::_create<float>('c', {bS, iC, kH, kW, oH, oW});

    auto outputPermuted = NDArrayFactory::_create<float>('c', {bS, oH, oW, iC, kH, kW});
    outputPermuted.permutei({0, 3, 4, 5, 1, 2});

    nd4j::ops::im2col op;

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&input}, {&output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // outputPermuted.printShapeInfo("permuted shape");

    auto permStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&input}, {&outputPermuted}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto permEnd = std::chrono::system_clock::now();
    auto permTime = std::chrono::duration_cast<std::chrono::microseconds> (permEnd - permStart).count();


    auto legacyStart = std::chrono::system_clock::now();

    float extra[] = {(float)kH, (float)kW, (float)sH, (float)sW, (float)pH, (float)pW, (float)dH, (float)dW, 0.f, 0.f};
    for (int e = 0; e < iterations; e++) {
        input.applyTransform(transform::Im2col, &output, extra);
    }

    auto legacyEnd = std::chrono::system_clock::now();
    auto legacyTime = std::chrono::duration_cast<std::chrono::microseconds> (legacyEnd - legacyStart).count();


    auto legacyPermStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        input.applyTransform(transform::Im2col, &outputPermuted, extra);
    }

    auto legacyPermEnd = std::chrono::system_clock::now();
    auto legacyPermTime = std::chrono::duration_cast<std::chrono::microseconds> (legacyPermEnd - legacyPermStart).count();


    NativeOps nativeOps;

    Nd4jLong iArgs[] = {kH, kW, sH, sW, pH, pW, dH, dW, 0};
    Nd4jPointer inputBuffers[] = {input.buffer()};
    Nd4jPointer inputShapes[] = {input.shapeInfo()};

    Nd4jPointer outputBuffers[] = {output.buffer()};
    Nd4jPointer outputShapes[] = {output.shapeInfo()};

    auto javaStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        nativeOps.execCustomOp(nullptr, op.getOpHash(), inputBuffers, inputShapes, 1, outputBuffers, outputShapes, 1, nullptr, 0, iArgs, 9, false);
    }

    auto javaEnd = std::chrono::system_clock::now();
    auto javaTime = std::chrono::duration_cast<std::chrono::microseconds> (javaEnd - javaStart).count();


    Nd4jPointer outputPermBuffers[] = {outputPermuted.buffer()};
    Nd4jPointer outputPermShapes[] = {outputPermuted.shapeInfo()};

    auto javaPermStart = std::chrono::system_clock::now();


    for (int e = 0; e < iterations; e++) {
        nativeOps.execCustomOp(nullptr, op.getOpHash(), inputBuffers, inputShapes, 1, outputPermBuffers, outputPermShapes, 1, nullptr, 0, iArgs, 9, false);
    }

    auto javaPermEnd = std::chrono::system_clock::now();
    auto javaPermTime = std::chrono::duration_cast<std::chrono::microseconds> (javaPermEnd - javaPermStart).count();

    // nd4j_printf("New time: %lld us;\n", outerTime / iterations);
    // nd4j_printf("Permuted time: %lld us;\n", permTime / iterations);
    // nd4j_printf("Legacy time: %lld us;\n", legacyTime / iterations);
    // nd4j_printf("Legacy Permuted time: %lld us;\n", legacyPermTime / iterations);
    // nd4j_printf("Java time: %lld us;\n", javaTime / iterations);
    // nd4j_printf("Java Permuted time: %lld us;\n", javaPermTime / iterations);
}

TEST_F(PlaygroundTests, Test_Im2Col_2) {
    auto input = NDArrayFactory::_create<float>('c', {16, 3, 224, 224});
    auto output = NDArrayFactory::_create<float>('c', {16, 3, 11, 11, 55, 55});

    auto outputPermuted = NDArrayFactory::_create<float>('c', {16, 55, 55, 3, 11, 11});
    outputPermuted.permutei({0, 3, 4, 5, 1, 2});

    nd4j::ops::im2col op;

    Nd4jLong iArgs[] = {11, 11, 4, 4, 2, 2, 1, 1, 0};
    Nd4jPointer inputBuffers[] = {input.buffer()};
    Nd4jPointer inputShapes[] = {input.shapeInfo()};

    Nd4jPointer outputPermBuffers[] = {outputPermuted.buffer()};
    Nd4jPointer outputPermShapes[] = {outputPermuted.shapeInfo()};

    NativeOps nativeOps;

    nativeOps.execCustomOp(nullptr, op.getOpHash(), inputBuffers, inputShapes, 1, outputPermBuffers, outputPermShapes, 1, nullptr, 0, iArgs, 9, false);
}

TEST_F(PlaygroundTests, Test_Col2Im_1) {
    
    int bS=16, iH=224,iW=224,  iC=3,oC=3,  kH=11,kW=11,  sH=4,sW=4,  pH=2,pW=2,  dH=1,dW=1;    
    int        oH=55, oW=55;
    int iterations = 1;

    auto input = NDArrayFactory::_create<float>('c', {bS, iC, kH, kW, oH, oW});
    auto output = NDArrayFactory::_create<float>('c', {bS, iC, iH, iW});
    
    auto inputPermuted = NDArrayFactory::_create<float>('c', {bS, oH, oW, iC, kH, kW});
    inputPermuted.permutei({0, 3, 4, 5, 1, 2});
    auto outputPermuted = NDArrayFactory::_create<float>('c', {bS, iH, iW, iC});
    outputPermuted.permutei({0, 3, 1, 2});

    input = 10.;
    output = 2.;

    inputPermuted = 10.;
    outputPermuted = 2.;

    nd4j::ops::col2im op;

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&input}, {&output}, {}, {sH, sW, pH, pW, iH, iW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    auto permStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&inputPermuted}, {&outputPermuted}, {}, {sH, sW, pH, pW, iH, iW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto permEnd = std::chrono::system_clock::now();
    auto permTime = std::chrono::duration_cast<std::chrono::microseconds> (permEnd - permStart).count();

    // nd4j_printf("C-order  time: %lld us;\n", outerTime / iterations);
    // nd4j_printf("Permuted time: %lld us;\n", permTime / iterations);    
}

TEST_F(PlaygroundTests, Test_Im2Col_3) {
    
    int bS=16, iH=224,iW=224,  iC=3,oC=3,  kH=11,kW=11,  sH=4,sW=4,  pH=2,pW=2,  dH=1,dW=1;    
    int        oH=55, oW=55;
    int iterations = 1;

    auto output = NDArrayFactory::_create<float>('c', {bS, iC, kH, kW, oH, oW});
    auto input = NDArrayFactory::_create<float>('c', {bS, iC, iH, iW});
    
    auto outputPermuted = NDArrayFactory::_create<float>('c', {bS, oH, oW, iC, kH, kW});
    outputPermuted.permutei({0, 3, 4, 5, 1, 2});
    auto inputPermuted = NDArrayFactory::_create<float>('c', {bS, iH, iW, iC});
    inputPermuted.permutei({0, 3, 1, 2});

    input = 10.;
    output = 2.;

    inputPermuted = 10.;
    outputPermuted = 2.;

    nd4j::ops::im2col op;

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&input}, {&output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    auto permStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++) {
        auto result = op.execute({&inputPermuted}, {&outputPermuted}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, 0});
        ASSERT_EQ(Status::OK(), result);
    }

    auto permEnd = std::chrono::system_clock::now();
    auto permTime = std::chrono::duration_cast<std::chrono::microseconds> (permEnd - permStart).count();

    // nd4j_printf("C-order  time: %lld us;\n", outerTime / iterations);
    // nd4j_printf("Permuted time: %lld us;\n", permTime / iterations);    
}


TEST_F(PlaygroundTests, loop_test_1) {

    auto f = NDArrayFactory::_create<float>('c', {2}, {5000, 10000});
    nd4j::ops::randomuniform op;

    auto result = op.execute({&f}, {-1.0f, 1.0f}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto array = result->at(0);

    auto buffer = array->buffer();
    int cnt = 0;
    int iterations = 1;

    //nd4j_printf("Array length: %lld\n", array->lengthOf());

    int length = (int) array->lengthOf();
    int span = (int) (array->lengthOf() / 6) + 8;

    NativeOps ops;

    auto t = new int[1000000];




    FloatBits fb;
    float threshold = 0.99f;
    fb.f_ = threshold;
    int le = ops.estimateThresholdFloat(nullptr, reinterpret_cast<void *>(array->buffer()), static_cast<int>(array->lengthOf()), threshold);

    t[0] = le;
    t[1] = length;
    t[2] = fb.i_;

    //nd4j_printf("number of elements: [%i]\n", le);

    long permTime = 0;

    for (int x = 0; x < iterations; x++) {
        auto permStart = std::chrono::system_clock::now();
        ops.estimateThresholdFloat(nullptr, reinterpret_cast<void *>(array->buffer()), static_cast<int>(array->lengthOf()), threshold);
        TypeCast::convertToThreshold<float>(nullptr, buffer, array->lengthOf(), t);

        /*
#pragma omp parallel reduction(+:cnt)
        {
            int tid = omp_get_thread_num();
            int start = span * tid;
            int stop = span * (tid + 1);
            if (stop > length)
                stop = length;

#pragma omp simd
            for (int e = start; e < stop; e++) {
                auto v = fabsf(buffer[e]);
                if (v >= 0.995f)
                    cnt++;
            }
        }
         */
/*
#pragma omp parallel for simd reduction(+:cnt)
        for (int e = 0; e < length; e++) {
            auto v = fabsf(buffer[e]);
            if (v >= 0.995f)
                cnt++;
        }
        */

        auto permEnd = std::chrono::system_clock::now();
        permTime += std::chrono::duration_cast<std::chrono::microseconds> (permEnd - permStart).count();
    }



    nd4j_printf("Permuted time: %lld us; Counter: %i;\n", permTime / iterations, cnt);

    delete result;
    delete[] t;
}

//////////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, ndarray_tile_test1) {

    auto x = NDArrayFactory::_create<float>('c', {20, 30});
    auto exp = NDArrayFactory::_create<float>('c', {2,40,60});

    auto timeStart = std::chrono::system_clock::now();
    auto tiled = x.tile({2,2,2});
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    // nd4j_printf("c-order time: %d;\n", time);
    
    ASSERT_TRUE(tiled.isSameShape(&exp)); 
}


//////////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, ndarray_tile_test2) {

    auto x = NDArrayFactory::_create<float>('f', {20, 30});
    auto exp = NDArrayFactory::_create<float>('f', {2,40,60});

    auto timeStart = std::chrono::system_clock::now();
    auto tiled = x.tile({2,2,2});
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    // nd4j_printf("f-order time: %d;\n", time);
    
    ASSERT_TRUE(tiled.isSameShape(&exp)); 
}
