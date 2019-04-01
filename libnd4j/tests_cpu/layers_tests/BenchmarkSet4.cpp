/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by Alex Black - 02/03/2019
//

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>
#include <ops/ops.h>
#include <OmpLaunchHelper.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/LegacyRandomOp.h>

using namespace nd4j;
using namespace nd4j::graph;

class BenchmarkSet4 : public testing::Test {
public:

    BenchmarkSet4() {
        printf("\n");
        fflush(stdout);
    }
};

#define PARAMETRIC_D() [&] (Parameters &p) -> Context*

TEST_F(BenchmarkSet4, GemmBenchmarksRegular) {
    BenchmarkHelper helper;

    for (int o = 0; o <= 1; o++) {
        char resultOrder = (o == 0 ? 'f' : 'c');
        for (int tA = 0; tA <= 1; tA++) {
            for (int tB = 0; tB <= 1; tB++) {


                IntPowerParameters pa("a", 2, 3, 12, 3);          //2^3=8, ..., 2^12=4096   ->  4 elements
                IntPowerParameters pb("b", 2, 4, 12, 4);          //2^4=16, ..., 2^12=4096   ->  3 elements
                IntPowerParameters pc("c", 2, 3, 12, 3);          //2^3=8, ..., 2^12=4096   ->  4 elements

                ParametersBatch b({&pa, &pb, &pc});

                auto generator = PARAMETRIC_XYZ() {
                    auto a = p.getIntParam("a");
                    auto b = p.getIntParam("b");
                    auto c = p.getIntParam("c");
                    std::vector<Nd4jLong> shapeA;
                    std::vector<Nd4jLong> shapeB;
                    if (tA) {
                        shapeA = {b, a};
                    } else {
                        shapeA = {a, b};
                    }
                    if (tB) {
                        shapeB = {c, b};
                    } else {
                        shapeB = {b, c};
                    }
                    auto A = NDArrayFactory::create_<float>('c', shapeA);
                    auto B = NDArrayFactory::create_<float>('c', shapeB);
                    auto C = NDArrayFactory::create_<float>(resultOrder, {a, c});

                    x.push_back(A);
                    y.push_back(B);
                    z.push_back(C);
                };

                std::string n;
                n += "Gemm - tA=";
                n += std::to_string(tA);
                n += ", tB=";
                n += std::to_string(tB);
                n += ", cOrder=";
                n += resultOrder;

                MatrixBenchmark mb(1.0, 0.0, tA == 0 ? false : true, tB == 0 ? false : true, n);

                helper.runOperationSuit(&mb, generator, b, n.c_str());
            }
        }
    }
}

TEST_F(BenchmarkSet4, GemmBenchmarksIrregular) {
    BenchmarkHelper helper;

    //Basically the same as above, but with irregular shapes (not multiples of 8, etc)

    for (int tA = 0; tA <= 1; tA++) {
        for (int tB = 0; tB <= 1; tB++) {
            IntParameters d("d", 1020, 1028, 1);     //1020, 1021, ..., 1028
            ParametersBatch dim({&d});

            //Vary A.rows:
            auto generator = PARAMETRIC_XYZ() {
                auto a = p.getIntParam("d");
                auto b = 1024;
                auto c = 1024;
                std::vector<Nd4jLong> shapeA;
                std::vector<Nd4jLong> shapeB;
                if (tA) {
                    shapeA = {b, a};
                } else {
                    shapeA = {a, b};
                }
                if (tB) {
                    shapeB = {c, b};
                } else {
                    shapeB = {b, c};
                }
                auto A = NDArrayFactory::create_<float>('c', shapeA);
                auto B = NDArrayFactory::create_<float>('c', shapeB);
                auto C = NDArrayFactory::create_<float>('f', {a, c});

                x.push_back(A);
                y.push_back(B);
                z.push_back(C);
            };

            std::string n;
            n += "Gemm (a.rows) - tA=";
            n += std::to_string(tA);
            n += ", tB=";
            n += std::to_string(tB);

            MatrixBenchmark mb(1.0, 0.0, tA, tB, n);

            helper.runOperationSuit(&mb, generator, dim, n.c_str());

            //Vary A.columns / B.rows
            auto generator2 = PARAMETRIC_XYZ() {
                auto a = 1024;
                auto b = p.getIntParam("d");
                auto c = 1024;
                std::vector<Nd4jLong> shapeA;
                std::vector<Nd4jLong> shapeB;
                if (tA) {
                    shapeA = {b, a};
                } else {
                    shapeA = {a, b};
                }
                if (tB) {
                    shapeB = {c, b};
                } else {
                    shapeB = {b, c};
                }
                auto A = NDArrayFactory::create_<float>('c', shapeA);
                auto B = NDArrayFactory::create_<float>('c', shapeB);
                auto C = NDArrayFactory::create_<float>('f', {a, c});

                x.push_back(A);
                y.push_back(B);
                z.push_back(C);
            };

            std::string n2;
            n2 += "Gemm (a.columns) - tA=";
            n2 += std::to_string(tA);
            n2 += ", tB=";
            n2 += std::to_string(tB);

            MatrixBenchmark mb2(1.0, 0.0, tA, tB, n2);

            helper.runOperationSuit(&mb2, generator2, dim, n2.c_str());

            //Vary A.columns / B.rows
            auto generator3 = PARAMETRIC_XYZ() {
                auto a = 1024;
                auto b = 1024;
                auto c = p.getIntParam("d");
                std::vector<Nd4jLong> shapeA;
                std::vector<Nd4jLong> shapeB;
                if (tA) {
                    shapeA = {b, a};
                } else {
                    shapeA = {a, b};
                }
                if (tB) {
                    shapeB = {c, b};
                } else {
                    shapeB = {b, c};
                }
                auto A = NDArrayFactory::create_<float>('c', shapeA);
                auto B = NDArrayFactory::create_<float>('c', shapeB);
                auto C = NDArrayFactory::create_<float>('f', {a, c});

                x.push_back(A);
                y.push_back(B);
                z.push_back(C);
            };

            std::string n3;
            n3 += "Gemm (b.columns) - tA=";
            n3 += std::to_string(tA);
            n3 += ", tB=";
            n3 += std::to_string(tB);

            MatrixBenchmark mb3(1.0, 0.0, tA, tB, n);

            helper.runOperationSuit(&mb3, generator3, dim, n3.c_str());
        }
    }
}

TEST_F(BenchmarkSet4, GemmBenchmarksMinibatchSize) {
    BenchmarkHelper helper;


    PredefinedParameters size("size", {128, 256, 512, 1024, 2048, 3072});
    //PredefinedParameters mb("mb", {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048});
    IntParameters mb("mb", 1, 2048, 1);

    ParametersBatch b({&size, &mb});

    auto generator = PARAMETRIC_XYZ() {
        auto m = p.getIntParam("mb");
        auto s = p.getIntParam("size");
        std::vector<Nd4jLong> shapeA;
        std::vector<Nd4jLong> shapeB;
        auto A = NDArrayFactory::create_<float>('c', {m, s});
        auto B = NDArrayFactory::create_<float>('c', {s, s});
        auto C = NDArrayFactory::create_<float>('c', {m, s});

        x.push_back(A);
        y.push_back(B);
        z.push_back(C);
    };

    std::string n;
    n += "gemm";

    MatrixBenchmark benchmark(1.0, 0.0, false , false, n);

    helper.runOperationSuit(&benchmark, generator, b, "Gemm - Benchmark Minibatch Sizes");
}

TEST_F(BenchmarkSet4, GemmBenchmarksMinibatchSize_CCF) {
    BenchmarkHelper helper;


    PredefinedParameters size("size", {128, 256, 512, 1024, 2048, 3072});
    //PredefinedParameters mb("mb", {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048});
    IntParameters mb("mb", 1, 2048, 1);

    ParametersBatch b({&size, &mb});

    auto generator = PARAMETRIC_XYZ() {
        auto m = p.getIntParam("mb");
        auto s = p.getIntParam("size");
        std::vector<Nd4jLong> shapeA;
        std::vector<Nd4jLong> shapeB;
        auto A = NDArrayFactory::create_<float>('c', {m, s});
        auto B = NDArrayFactory::create_<float>('c', {s, s});
        auto C = NDArrayFactory::create_<float>('f', {m, s});

        x.push_back(A);
        y.push_back(B);
        z.push_back(C);
    };

    std::string n;
    n += "gemm";

    MatrixBenchmark benchmark(1.0, 0.0, false , false, n);

    helper.runOperationSuit(&benchmark, generator, b, "Gemm - Benchmark Minibatch Sizes - CCF");
}

TEST_F(BenchmarkSet4, GemmBenchmarksLayerSizes) {
    BenchmarkHelper helper;


    //IntParameters size("size", 32, 3072, 32);
    IntParameters size("size", 8, 3072, 1);
    PredefinedParameters mb("mb", {1, 8, 32, 64, 128, 256, 512});

    ParametersBatch b({&size, &mb});

    auto generator = PARAMETRIC_XYZ() {
        auto m = p.getIntParam("mb");
        auto s = p.getIntParam("size");
        std::vector<Nd4jLong> shapeA;
        std::vector<Nd4jLong> shapeB;
        auto A = NDArrayFactory::create_<float>('c', {m, s});
        auto B = NDArrayFactory::create_<float>('c', {s, s});
        auto C = NDArrayFactory::create_<float>('c', {m, s});

        x.push_back(A);
        y.push_back(B);
        z.push_back(C);
    };

    std::string n;
    n += "gemm";

    MatrixBenchmark benchmark(1.0, 0.0, false , false, n);

    helper.runOperationSuit(&benchmark, generator, b, "Gemm - Benchmark Layer Sizes");
}

TEST_F(BenchmarkSet4, GemmBenchmarksLayerSizes_CCF) {
    BenchmarkHelper helper;


    //IntParameters size("size", 32, 3072, 32);
    IntParameters size("size", 8, 3072, 1);
    PredefinedParameters mb("mb", {1, 8, 32, 64, 128, 256, 512});

    ParametersBatch b({&size, &mb});

    auto generator = PARAMETRIC_XYZ() {
        auto m = p.getIntParam("mb");
        auto s = p.getIntParam("size");
        std::vector<Nd4jLong> shapeA;
        std::vector<Nd4jLong> shapeB;
        auto A = NDArrayFactory::create_<float>('c', {m, s});
        auto B = NDArrayFactory::create_<float>('c', {s, s});
        auto C = NDArrayFactory::create_<float>('f', {m, s});

        x.push_back(A);
        y.push_back(B);
        z.push_back(C);
    };

    std::string n;
    n += "gemm";

    MatrixBenchmark benchmark(1.0, 0.0, false , false, n);

    helper.runOperationSuit(&benchmark, generator, b, "Gemm - Benchmark Layer Sizes");
}


TEST_F(BenchmarkSet4, RNGBenchmarks) {
    BenchmarkHelper helper;

    //Uniform, gaussian and bernoulli RNG generation

    IntPowerParameters length("length", 2, 4, 30, 3);      //2^8 to 2^30 in steps of 3

    ParametersBatch batch({&length});

    auto gen01 = PARAMETRIC_D() {
        auto ctx = new Context(1);
        ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>('c', {2},
                                                                {1, p.getIntParam("length")}));   //Shape as NDArray
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {1, p.getIntParam("length")}));
        double *d = new double[2];
        d[0] = 0.0;
        d[1] = 1.0;
        ctx->setTArguments(d, 2);
        delete[] d;
        return ctx;
    };

    auto gen05 = PARAMETRIC_D() {
        auto ctx = new Context(1);
        ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>('c', {2},
                                                                {1, p.getIntParam("length")}));   //Shape as NDArray
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {1, p.getIntParam("length")}));
        double *d = new double[1];
        d[0] = 0.5;
        ctx->setTArguments(d, 1);
        delete[] d;
        return ctx;
    };

    nd4j::ops::LegacyRandomOp unif(random::UniformDistribution);
    DeclarableBenchmark dbU(unif, "uniform");
    helper.runOperationSuit(&dbU, gen01, batch, "Uniform Distribution");

    nd4j::ops::LegacyRandomOp gaussian(random::GaussianDistribution);
    DeclarableBenchmark dbG(gaussian, "gaussian");
    helper.runOperationSuit(&dbG, gen01, batch, "Gaussian Distribution");

    nd4j::ops::LegacyRandomOp trunc(random::TruncatedNormalDistribution);
    DeclarableBenchmark dbTU(unif, "trunc.norm");
    helper.runOperationSuit(&dbTU, gen01, batch, "Truncated Normal Distribution");

    nd4j::ops::LegacyRandomOp ln(random::LogNormalDistribution);
    DeclarableBenchmark dbLN(ln, "uniform");
    helper.runOperationSuit(&dbLN, gen01, batch, "Log Normal Distribution");

    nd4j::ops::LegacyRandomOp bernoulli(random::BernoulliDistribution);
    DeclarableBenchmark dbB(bernoulli, "bernoulli");
    helper.runOperationSuit(&dbB, gen05, batch, "Bernoulli Distribution");

    nd4j::ops::LegacyRandomOp dropout(random::BernoulliDistribution);
    DeclarableBenchmark dbD(dropout, "dropout");
    helper.runOperationSuit(&dbD, gen05, batch, "Dropout");

    /*
    auto generator = PARAMETRIC_XZ() {
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
        x.push_back(arr);
        if(p.getIntParam("inplace") == 1){
            z.push_back(arr);
        } else {
            z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
        }
    };

    nd4j::graph::RandomGenerator r0(12345);
    double* extraArgs = new double[2];
    extraArgs[0] = 0.0;
    extraArgs[1] = 1.0;

    std::string name("uniform");

    RandomBenchmark rbUniform(random::Ops::UniformDistribution, name, &r0, extraArgs);
    helper.runOperationSuit(&rbUniform, generator, batch, "Uniform");


    std::string name2("gaussian");
    RandomBenchmark rbGaussian(random::Ops::GaussianDistribution, name2, &r0, extraArgs);
    helper.runOperationSuit(&rbGaussian, generator, batch, "Gaussian");


    std::string name3("trunc.norm");
    RandomBenchmark rbTNorm(random::Ops::TruncatedNormalDistribution, name3, &r0, extraArgs);
    helper.runOperationSuit(&rbTNorm, generator, batch, "Truncated Normal");

    std::string nameLN("lognormal");
    RandomBenchmark rbLogNormal(random::Ops::LogNormalDistribution, nameLN, &r0, extraArgs);
    helper.runOperationSuit(&rbLogNormal, generator, batch, "Log Normal");

    delete[] extraArgs;

    double* extraArgs2 = new double[1];
    extraArgs2[0] = 0.5;
    std::string name4("bernoulli");
    RandomBenchmark rbBernoulli(random::Ops::BernoulliDistribution, name4, &r0, extraArgs2);
    helper.runOperationSuit(&rbBernoulli, generator, batch, "Bernoulli Distribution");

    std::string name5("dropout");
    RandomBenchmark rbDropout(random::Ops::DropOutInverted, name5, &r0, extraArgs2);
    helper.runOperationSuit(&rbDropout, generator, batch, "Dropout (Inverted)");

    std::string nameEx("exponential");
    RandomBenchmark rbExp(random::Ops::ExponentialDistribution, nameEx, &r0, extraArgs2);
    helper.runOperationSuit(&rbDropout, generator, batch, "Exponential");

    delete[] extraArgs2;
     */



}


TEST_F(BenchmarkSet4, Conv2dBenchmarks) {
    BenchmarkHelper helper;

    //Convolution2D op
    BoolParameters nhwc("nhwc");
    PredefinedParameters k("k", {2, 3, 5});
    PredefinedParameters c("c", {3, 32, 128});
    PredefinedParameters hw("hw", {32, 128});

    ParametersBatch batch({&nhwc, &k, &c, &hw});
    nd4j::ops::conv2d conv2d;
    DeclarableBenchmark benchmark(conv2d, "conv2d");

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int n = p.getIntParam("nhwc");
        int hw = p.getIntParam("hw");
        int khw = p.getIntParam("k");

        if (n == 0) {
            auto input = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
            auto output = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
        } else {
            auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
            auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
        }

        auto b = NDArrayFactory::create_<float>('c', {p.getIntParam("c")});
        auto w = NDArrayFactory::create_<float>('c', {khw, khw, p.getIntParam("c"),
                                                      p.getIntParam("c")});   // [kH, kW, iC, oC] always

        ctx->setInputArray(1, w);
        ctx->setInputArray(2, b);

        Nd4jLong *args = new Nd4jLong[10];
        args[0] = args[1] = khw; //Kernel
        args[2] = args[3] = 1;//Stride
        args[4] = args[5] = 0;  //Pad
        args[6] = args[7] = 1;  //Dilation
        args[8] = 1;     //SAME
        args[9] = n;//0-nchw, 1=nhwc
        ctx->setIArguments(args, 10);
        delete[] args;

        return ctx;
    };

    helper.runOperationSuit(&benchmark, generator, batch, "Conv2d Operation");
}


TEST_F(BenchmarkSet4, Pool2dBenchmarks) {
    BenchmarkHelper helper;

    //Convolution2D op
    BoolParameters nhwc("nhwc");
    PredefinedParameters k("k", {2, 3, 5});
    PredefinedParameters c("c", {3, 32, 128});
    PredefinedParameters hw("hw", {32, 128});

    ParametersBatch batch({&nhwc, &k, &c, &hw});

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int n = p.getIntParam("nhwc");
        int hw = p.getIntParam("hw");
        int khw = p.getIntParam("k");

        if (n == 0) {
            auto input = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
            auto output = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
        } else {
            auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
            auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
        }

        Nd4jLong *args = new Nd4jLong[11];
        args[0] = args[1] = khw; //Kernel
        args[2] = args[3] = 1;//Stride
        args[4] = args[5] = 0;  //Pad
        args[6] = args[7] = 1;  //Dilation
        args[8] = 1;     //SAME
        args[9] = 0;     //Divisor mode - 0 = exclude padding in divisor
        args[10] = n;//0-nchw, 1=nhwc
        ctx->setIArguments(args, 11);
        delete[] args;

        return ctx;
    };

    nd4j::ops::avgpool2d avgpool2d;
    DeclarableBenchmark benchmark1(avgpool2d, "avgpool");
    helper.runOperationSuit(&benchmark1, generator, batch, "Average Pooling 2d Operation");

    nd4j::ops::maxpool2d maxpool2d;
    DeclarableBenchmark benchmark2(maxpool2d, "maxpool");
    helper.runOperationSuit(&benchmark2, generator, batch, "Max Pooling 2d Operation");
}

TEST_F(BenchmarkSet4, BatchNormBenchmarks) {
    BenchmarkHelper helper;

    //Convolution2D op
    BoolParameters nhwc("nhwc");
    PredefinedParameters c("c", {3, 32, 128});
    PredefinedParameters hw("hw", {32, 128});

    ParametersBatch batch({&nhwc, &c, &hw});

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int n = p.getIntParam("nhwc");
        int hw = p.getIntParam("hw");
        int ch = p.getIntParam("c");

        Nd4jLong *args = new Nd4jLong[3];
        args[0] = args[1] = 1; //apply scale and offset
        if (n == 0) {
            auto input = NDArrayFactory::create_<float>('c', {32, ch, hw, hw});
            auto output = NDArrayFactory::create_<float>('c', {32, ch, hw, hw});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
            args[2] = 1;    //axis
        } else {
            auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, ch});
            auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, ch});
            ctx->setInputArray(0, input);
            ctx->setOutputArray(0, output);
            args[2] = 3;    //axis
        }
        ctx->setIArguments(args, 3);
        delete[] args;

        ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {ch}));   //mean
        auto v = NDArrayFactory::create_<float>('c', {ch});
        v->assign(1.0f);
        ctx->setInputArray(2, v);   //variance
        auto g = NDArrayFactory::create_<float>('c', {ch});
        g->assign(1.0);
        ctx->setInputArray(3, g);   //gamma
        auto b = NDArrayFactory::create_<float>('c', {ch});
        b->assign(1.0);
        ctx->setInputArray(4, b);   //beta

        double *targs = new double[1];
        targs[0] = 1e-5;
        ctx->setTArguments(targs, 1);
        delete[] targs;

        return ctx;
    };

    nd4j::ops::batchnorm_new batchnorm;
    DeclarableBenchmark benchmark(batchnorm, "batchnorm");
    helper.runOperationSuit(&benchmark, generator, batch, "Batch Normalization");
}


TEST_F(BenchmarkSet4, LSTMBenchmarks) {
    BenchmarkHelper helper;

    BoolParameters format("format");    //0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]
    PredefinedParameters mb("mb", {1, 8, 64});
    PredefinedParameters nInOut("nInOut", {32, 256, 1024});

    ParametersBatch batch({&format, &mb, &nInOut});
    nd4j::ops::lstmBlock lstmBlock;
    DeclarableBenchmark benchmark(lstmBlock, "lstm");

    int seqLength = 64;

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int f = p.getIntParam("format");
        int m = p.getIntParam("mb");
        int n = p.getIntParam("nInOut");

        Nd4jLong l = 0;
        ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>(l));  //Max TS length (unused)


        if (f == 0) {
            //TNS format
            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}));     //x
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //i
            ctx->setOutputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //c
            ctx->setOutputArray(2, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //f
            ctx->setOutputArray(3, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //o
            ctx->setOutputArray(4, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //z
            ctx->setOutputArray(5, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //h
            ctx->setOutputArray(6, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //y
        } else {
            //NST format
            ctx->setInputArray(1, NDArrayFactory::create_<float>('f', {m, n, seqLength}));     //x
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //i
            ctx->setOutputArray(1, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //c
            ctx->setOutputArray(2, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //f
            ctx->setOutputArray(3, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //o
            ctx->setOutputArray(4, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //z
            ctx->setOutputArray(5, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //h
            ctx->setOutputArray(6, NDArrayFactory::create_<float>('f', {m, n, seqLength}));    //y
        }

        auto cLast = NDArrayFactory::create_<float>('c', {m, n});
        auto yLast = NDArrayFactory::create_<float>('c', {m, n});
        auto W = NDArrayFactory::create_<float>('c', {2 * n, 4 * n});
        auto Wci = NDArrayFactory::create_<float>('c', {n});
        auto Wcf = NDArrayFactory::create_<float>('c', {n});
        auto Wco = NDArrayFactory::create_<float>('c', {n});
        auto b = NDArrayFactory::create_<float>('c', {4 * n});

        ctx->setInputArray(2, cLast);
        ctx->setInputArray(3, yLast);
        ctx->setInputArray(4, W);
        ctx->setInputArray(5, Wci);
        ctx->setInputArray(6, Wcf);
        ctx->setInputArray(7, Wco);
        ctx->setInputArray(8, b);

        Nd4jLong *iargs = new Nd4jLong[2];
        iargs[0] = 0;   //No peephole
        iargs[1] = f;
        ctx->setIArguments(iargs, 2);
        delete[] iargs;
        double *targs = new double[2];
        targs[0] = 1.0; //forget bias
        targs[1] = 0.0; //cell clipping value
        ctx->setTArguments(targs, 2);
        delete[] targs;
        return ctx;
    };

    helper.runOperationSuit(&benchmark, generator, batch, "LSTMBlock");
}



TEST_F(BenchmarkSet4, LSTMBenchmarks_DebugTNS) {
    BenchmarkHelper helper(5,10);

    PredefinedParameters mb("mb", {1, 8, 64});
    PredefinedParameters nInOut("nInOut", {32, 256, 1024});

    ParametersBatch batch({&mb, &nInOut});
    nd4j::ops::lstmBlock lstmBlock;
    DeclarableBenchmark benchmark(lstmBlock, "lstm");

    int seqLength = 64;

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int m = p.getIntParam("mb");
        int n = p.getIntParam("nInOut");

        Nd4jLong l = 0;
        ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>(l));  //Max TS length (unused)


        //TNS format
        ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}));     //x
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //i
        ctx->setOutputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //c
        ctx->setOutputArray(2, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //f
        ctx->setOutputArray(3, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //o
        ctx->setOutputArray(4, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //z
        ctx->setOutputArray(5, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //h
        ctx->setOutputArray(6, NDArrayFactory::create_<float>('c', {seqLength, m, n}));    //y

        auto cLast = NDArrayFactory::create_<float>('c', {m, n});
        auto yLast = NDArrayFactory::create_<float>('c', {m, n});
        auto W = NDArrayFactory::create_<float>('c', {2 * n, 4 * n});
        auto Wci = NDArrayFactory::create_<float>('c', {n});
        auto Wcf = NDArrayFactory::create_<float>('c', {n});
        auto Wco = NDArrayFactory::create_<float>('c', {n});
        auto b = NDArrayFactory::create_<float>('c', {4 * n});

        ctx->setInputArray(2, cLast);
        ctx->setInputArray(3, yLast);
        ctx->setInputArray(4, W);
        ctx->setInputArray(5, Wci);
        ctx->setInputArray(6, Wcf);
        ctx->setInputArray(7, Wco);
        ctx->setInputArray(8, b);

        Nd4jLong *iargs = new Nd4jLong[2];
        iargs[0] = 0;   //No peephole
        iargs[1] = 0;   //TNS
        ctx->setIArguments(iargs, 2);
        delete[] iargs;
        double *targs = new double[2];
        targs[0] = 1.0; //forget bias
        targs[1] = 0.0; //cell clipping value
        ctx->setTArguments(targs, 2);
        delete[] targs;
        return ctx;
    };

    helper.runOperationSuit(&benchmark, generator, batch, "LSTMBlock");
}
