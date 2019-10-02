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
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <algorithm>

#ifdef _RELEASE
    int wIterations = 4;
    int rIterations = 20;
    int gemmRegularUpperPow = 11;
    int scalarBenchmarkPowLimit = 26;
    int transformBenchmarkPowLimit = 26;
    int intermediateTransformPowLimit = 22;
    int intermediateTransformPowLimit2 = 18;
    int pairwisePowLimit = 26;
    int heavyPowLimit = 22;
    int nonEwsPowLimit = 10;
    int reduceScalarPowLimit = 26;
    int stridedReductionPowLimit = 20;
    int mismatchedAssignPowLimit = 26;
    int gatherOpPowLimit = 18;
    int gatherOpPowLimit2 = 16;
    int gatherOpPowLimit3 = 12;
    int broadcastMatrixRankLimit = 5;
    int limit30 = 30;
    int limit26 = 26;
    int limit24 = 24;
    int limit22 = 22;
    int limit20 = 20;
    int limit18 = 18;
    int limit10 = 10;
    int limit5 = 5;
    int limit3 = 3;
#else
    int wIterations = 0;
    int rIterations = 1;
    int gemmRegularUpperPow = 7;
    int scalarBenchmarkPowLimit = 10;
    int transformBenchmarkPowLimit = 10;
    int intermediateTransformPowLimit = 10;
    int intermediateTransformPowLimit2 = 10;
    int pairwisePowLimit = 10;
    int heavyPowLimit = 10;
    int nonEwsPowLimit = 6;
    int reduceScalarPowLimit = 10;
    int stridedReductionPowLimit = 12;
    int mismatchedAssignPowLimit = 2;
    int gatherOpPowLimit = 10;
    int gatherOpPowLimit2 = 8;
    int gatherOpPowLimit3 = 8;
    int broadcastMatrixRankLimit = 3;
    int limit26 = 8;
    int limit24 = 8;
    int limit22 = 8;
    int limit20 = 8;
    int limit18 = 8;
    int limit10 = 4;
    int limit5 = 3;
    int limit3 = 1;
#endif

namespace nd4j {

    static std::string layerNormBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        BoolParameters nhwc("nhwc");    //0 = nchw

#ifdef _RELEASE
        int c = 32;
        int hw = 64;
#else
        int c = 3;
        int hw = 8;
#endif

        ParametersBatch batch({&nhwc});

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int n = p.getIntParam("nhwc");

            int axis;
            if (n == 0) {
                //nchw
                auto input = NDArrayFactory::create_<float>('c', {16, c, hw, hw});
                auto output = NDArrayFactory::create_<float>('c', {16, c, hw, hw});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
                axis = 1;
            } else {
                auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, c});
                auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, c});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
                axis = 3;
            }

            auto bias = NDArrayFactory::create_<float>('c', {c});
            ctx->setInputArray(1, bias, true);
            auto iargs = new Nd4jLong[1];
            iargs[0] = axis;
            ctx->setIArguments(iargs, 1);
            delete[] iargs;

            return ctx;
        };

        nd4j::ops::layer_norm layerNorm;
        DeclarableBenchmark benchmark(layerNorm, "layer norm");
        output += helper.runOperationSuit(&benchmark, generator, batch, "Layer Norm");

        return output;
    }


    static std::string maxPool3DBenchmark(){
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        BoolParameters ncdhw("ncdhw");  //1 = ndhwc
        ParametersBatch batch({&ncdhw});

        nd4j::ops::maxpool3dnew maxpool3Dnew;
        DeclarableBenchmark benchmark(maxpool3Dnew, "maxPool3d");

#ifdef _RELEASE
        int mb = 16;
        int chIn = 16;
        int chOut = 16;
        int dhw = 64;
#else
        int mb = 1;
        int chIn = 3;
        int chOut = 3;
        int dhw = 16;
#endif

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int format = p.getIntParam("ncdhw");

            //Set inputs and outputs
            //Same mode + stride 1: output is same shape as input
            if(format == 1) {
                //NDHWC
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {mb, dhw, dhw, dhw, chIn}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {mb, dhw, dhw, dhw, chIn}), true);
            } else {
                //NCDHW
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {mb, chIn, dhw, dhw, dhw}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {mb, chIn, dhw, dhw, dhw}), true);
            }

            auto iargs = new Nd4jLong[15];
            //Kernel, strides, padding, dilation - x3 each
            iargs[0] = 3;   //Kernel
            iargs[1] = 3;
            iargs[2] = 3;
            iargs[3] = 1;   //Stride
            iargs[4] = 1;
            iargs[5] = 1;
            iargs[6] = 0;   //Padding
            iargs[7] = 0;
            iargs[8] = 0;
            iargs[9] = 1;   //Dilation
            iargs[10] = 1;
            iargs[11] = 1;
            iargs[12] = 1;  //Same mode
            iargs[13] = 0;  //Unused for max
            iargs[14] = format; //0 = ncdhw
            ctx->setIArguments(iargs, 14);
            delete[] iargs;

            return ctx;
        };

        output += helper.runOperationSuit(&benchmark, generator, batch, "maxPool3d");
        return output;
    }


    static std::string conv3dBenchmark(){
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        BoolParameters ncdhw("ncdhw");  //1 = ndhwc
        ParametersBatch batch({&ncdhw});

        nd4j::ops::conv3dnew conv3Dnew;
        DeclarableBenchmark benchmark(conv3Dnew, "conv3d");

#ifdef _RELEASE
        int mb = 16;
        int chIn = 16;
        int chOut = 16;
        int dhw = 64;
#else
        int mb = 1;
        int chIn = 3;
        int chOut = 3;
        int dhw = 16;
#endif

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int format = p.getIntParam("ncdhw");

            //Set inputs and outputs
            //Same mode + stride 1: output is same shape as input
            if(format == 1) {
                //NDHWC
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {mb, dhw, dhw, dhw, chIn}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {mb, dhw, dhw, dhw, chIn}), true);
            } else {
                //NCDHW
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {mb, chIn, dhw, dhw, dhw}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {mb, chIn, dhw, dhw, dhw}), true);
            }

            //Weights and bias:
            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {3, 3, 3, chIn, chOut}), true);
            ctx->setInputArray(2, NDArrayFactory::create_<float>('c', {chOut}), true);


            auto iargs = new Nd4jLong[14];
            //Kernel, strides, padding, dilation - x3 each
            iargs[0] = 3;   //Kernel
            iargs[1] = 3;
            iargs[2] = 3;
            iargs[3] = 1;   //Stride
            iargs[4] = 1;
            iargs[5] = 1;
            iargs[6] = 0;   //Padding
            iargs[7] = 0;
            iargs[8] = 0;
            iargs[9] = 1;   //Dilation
            iargs[10] = 1;
            iargs[11] = 1;
            iargs[12] = 1;  //Same mode
            iargs[13] = format; //0 = ncdhw
            ctx->setIArguments(iargs, 14);
            delete[] iargs;

            return ctx;
        };

        output += helper.runOperationSuit(&benchmark, generator, batch, "CNN3D");
        return output;
    }


    static std::string lstmBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        BoolParameters format("format");    //0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]
#ifdef _RELEASE
        PredefinedParameters mb("mb", {1, 8, 64});
        PredefinedParameters nInOut("nInOut", {32, 256, 1024});
#else
        PredefinedParameters mb("mb", {1});
        PredefinedParameters nInOut("nInOut", {32});
#endif

        ParametersBatch batch({&format, &mb, &nInOut});
        nd4j::ops::lstmBlock lstmBlock;
        DeclarableBenchmark benchmark(lstmBlock, "lstm");

        int seqLength = 32;

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int f = p.getIntParam("format");
            int m = p.getIntParam("mb");
            int n = p.getIntParam("nInOut");

            Nd4jLong l = 0;
            ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>(l), true);  //Max TS length (unused)


            if (f == 0) {
                //TNS format
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);     //x
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //i
                ctx->setOutputArray(1, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //c
                ctx->setOutputArray(2, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //f
                ctx->setOutputArray(3, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //o
                ctx->setOutputArray(4, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //z
                ctx->setOutputArray(5, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //h
                ctx->setOutputArray(6, NDArrayFactory::create_<float>('c', {seqLength, m, n}), true);    //y
            } else {
                //NST format
                ctx->setInputArray(1, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);     //x
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //i
                ctx->setOutputArray(1, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //c
                ctx->setOutputArray(2, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //f
                ctx->setOutputArray(3, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //o
                ctx->setOutputArray(4, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //z
                ctx->setOutputArray(5, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //h
                ctx->setOutputArray(6, NDArrayFactory::create_<float>('f', {m, n, seqLength}), true);    //y
            }

            auto cLast = NDArrayFactory::create_<float>('c', {m, n});
            auto yLast = NDArrayFactory::create_<float>('c', {m, n});
            auto W = NDArrayFactory::create_<float>('c', {2 * n, 4 * n});
            auto Wci = NDArrayFactory::create_<float>('c', {n});
            auto Wcf = NDArrayFactory::create_<float>('c', {n});
            auto Wco = NDArrayFactory::create_<float>('c', {n});
            auto b = NDArrayFactory::create_<float>('c', {4 * n});

            ctx->setInputArray(2, cLast, true);
            ctx->setInputArray(3, yLast, true);
            ctx->setInputArray(4, W, true);
            ctx->setInputArray(5, Wci, true);
            ctx->setInputArray(6, Wcf, true);
            ctx->setInputArray(7, Wco, true);
            ctx->setInputArray(8, b, true);

            auto iargs = new Nd4jLong[2];
            iargs[0] = 0;   //No peephole
            iargs[1] = f;
            ctx->setIArguments(iargs, 2);
            delete[] iargs;

            auto targs = new double[2];
            targs[0] = 1.0; //forget bias
            targs[1] = 0.0; //cell clipping value
            ctx->setTArguments(targs, 2);
            delete[] targs;
            return ctx;
        };

        output += helper.runOperationSuit(&benchmark, generator, batch, "LSTMBlock");
        return output;
    }

    static std::string batchnormBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Convolution2D op
        BoolParameters nhwc("nhwc");
#ifdef _RELEASE
        PredefinedParameters c("c", {3, 32, 128});
        PredefinedParameters hw("hw", {32, 128});
#else
        PredefinedParameters c("c", {3});
        PredefinedParameters hw("hw", {16});
#endif

        ParametersBatch batch({&nhwc, &c, &hw});

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int n = p.getIntParam("nhwc");
            int hw = p.getIntParam("hw");
            int ch = p.getIntParam("c");

            auto args = new Nd4jLong[3];
            args[0] = args[1] = 1; //apply scale and offset
            if (n == 0) {
                auto input = NDArrayFactory::create_<float>('c', {32, ch, hw, hw});
                auto output = NDArrayFactory::create_<float>('c', {32, ch, hw, hw});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
                args[2] = 1;    //axis
            } else {
                auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, ch});
                auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, ch});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
                args[2] = 3;    //axis
            }
            ctx->setIArguments(args, 3);
            delete[] args;

            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {ch}), true);   //mean
            auto v = NDArrayFactory::create_<float>('c', {ch});
            v->assign(1.0f);
            ctx->setInputArray(2, v, true);   //variance
            auto g = NDArrayFactory::create_<float>('c', {ch});
            g->assign(1.0);
            ctx->setInputArray(3, g, true);   //gamma
            auto b = NDArrayFactory::create_<float>('c', {ch});
            b->assign(1.0);
            ctx->setInputArray(4, b, true);   //beta

            auto targs = new double[1];
            targs[0] = 1e-5;
            ctx->setTArguments(targs, 1);
            delete[] targs;

            return ctx;
        };

        nd4j::ops::batchnorm_new batchnorm;
        DeclarableBenchmark benchmark(batchnorm, "batchnorm");
        output += helper.runOperationSuit(&benchmark, generator, batch, "Batch Normalization");

        return output;
    }

    static std::string pool2dBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Convolution2D op
        BoolParameters nhwc("nhwc");
#ifdef _RELEASE
        PredefinedParameters k("k", {2, 3, 5});
        PredefinedParameters c("c", {3, 32, 128});
        PredefinedParameters hw("hw", {32, 128});
#else
        PredefinedParameters k("k", {2});
        PredefinedParameters c("c", {3});
        PredefinedParameters hw("hw", {8});
#endif

        ParametersBatch batch({&nhwc, &k, &c, &hw});

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int n = p.getIntParam("nhwc");
            int hw = p.getIntParam("hw");
            int khw = p.getIntParam("k");

            if (n == 0) {
                auto input = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
                auto output = NDArrayFactory::create_<float>('c', {32, p.getIntParam("c"), hw, hw});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            } else {
                auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
                auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            }

            auto args = new Nd4jLong[11];
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
        output += helper.runOperationSuit(&benchmark1, generator, batch, "Average Pooling 2d Operation");

        nd4j::ops::maxpool2d maxpool2d;
        DeclarableBenchmark benchmark2(maxpool2d, "maxpool");
        output += helper.runOperationSuit(&benchmark2, generator, batch, "Max Pooling 2d Operation");
        return output;
    }

    static std::string conv2dBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Convolution2D op
        BoolParameters nhwc("nhwc");
#ifdef _RELEASE
        PredefinedParameters k("k", {2, 3, 5});
        PredefinedParameters c("c", {3, 32, 128});
        PredefinedParameters hw("hw", {32, 128});
#else
        PredefinedParameters k("k", {2});
        PredefinedParameters c("c", {3});
        PredefinedParameters hw("hw", {8});
#endif
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
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            } else {
                auto input = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
                auto output = NDArrayFactory::create_<float>('c', {32, hw, hw, p.getIntParam("c")});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            }

            auto b = NDArrayFactory::create_<float>('c', {p.getIntParam("c")});
            auto w = NDArrayFactory::create_<float>('c', {khw, khw, p.getIntParam("c"), p.getIntParam("c")});   // [kH, kW, iC, oC] always

            ctx->setInputArray(1, w, true);
            ctx->setInputArray(2, b, true);

            auto args = new Nd4jLong[10];
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

        output += helper.runOperationSuit(&benchmark, generator, batch, "Conv2d Operation");
        return output;
    }

    static std::string rngBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);
        //Uniform, gaussian and bernoulli RNG generation

        IntPowerParameters length("length", 2, 4, scalarBenchmarkPowLimit, 3);      //2^8 to 2^30 in steps of 3

        ParametersBatch batch({&length});

        auto gen01 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>('c', {2},{1, p.getIntParam("length")}), true);   //Shape as NDArray
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {1, p.getIntParam("length")}), true);
            auto d = new double[2];
            d[0] = 0.0;
            d[1] = 1.0;
            ctx->setTArguments(d, 2);
            delete[] d;
            return ctx;
        };

        auto gen05 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>('c', {2},{1, p.getIntParam("length")}), true);   //Shape as NDArray
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {1, p.getIntParam("length")}), true);
            auto d = new double[1];
            d[0] = 0.5;
            ctx->setTArguments(d, 1);
            delete[] d;
            return ctx;
        };

        nd4j::ops::LegacyRandomOp unif(random::UniformDistribution);
        DeclarableBenchmark dbU(unif, "uniform");
        output += helper.runOperationSuit(&dbU, gen01, batch, "Uniform Distribution");

        nd4j::ops::LegacyRandomOp gaussian(random::GaussianDistribution);
        DeclarableBenchmark dbG(gaussian, "gaussian");
        output += helper.runOperationSuit(&dbG, gen01, batch, "Gaussian Distribution");

        nd4j::ops::LegacyRandomOp trunc(random::TruncatedNormalDistribution);
        DeclarableBenchmark dbTU(unif, "trunc.norm");
        output += helper.runOperationSuit(&dbTU, gen01, batch, "Truncated Normal Distribution");

        nd4j::ops::LegacyRandomOp ln(random::LogNormalDistribution);
        DeclarableBenchmark dbLN(ln, "uniform");
        output += helper.runOperationSuit(&dbLN, gen01, batch, "Log Normal Distribution");

        nd4j::ops::LegacyRandomOp bernoulli(random::BernoulliDistribution);
        DeclarableBenchmark dbB(bernoulli, "bernoulli");
        output += helper.runOperationSuit(&dbB, gen05, batch, "Bernoulli Distribution");

        nd4j::ops::LegacyRandomOp dropout(random::BernoulliDistribution);
        DeclarableBenchmark dbD(dropout, "dropout");
        output += helper.runOperationSuit(&dbD, gen05, batch, "Dropout");

        return output;
    }

    static std::string gemmIrregularBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Basically the same as above, but with irregular shapes (not multiples of 8, etc)

#ifdef _RELEASE
        int tAMax = 1;
        int tBMax = 1;
        int b = 1024;
        int c = 1024;
#else
        int tAMax = 1;
        int tBMax = 1;
        int b = 32;
        int c = 32;
#endif

        for (int tA = 0; tA <= tAMax; tA++) {
            for (int tB = 0; tB <= tBMax; tB++) {
                IntParameters d("d", 1020, 1028, 1);     //1020, 1021, ..., 1028
                ParametersBatch dim({&d});

                //Vary A.rows:
                auto generator = PARAMETRIC_XYZ() {
                    auto a = p.getIntParam("d");
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

                output += helper.runOperationSuit(&mb, generator, dim, n.c_str());

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

                output += helper.runOperationSuit(&mb2, generator2, dim, n2.c_str());

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

                output += helper.runOperationSuit(&mb3, generator3, dim, n3.c_str());
            }
        }

        return output;
    }

    static std::string batchGemmBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Rank 3 - [32,1024,1024]x[32,1024,1024]
        //Rank 4 - [4,8,1024,1024]x[4,8,1024,1024]

        IntParameters rank("rank", 3, 4, 1);

        ParametersBatch b({&rank});

        auto generator = PARAMETRIC_D() {
            auto rank = p.getIntParam("rank");
            std::vector<Nd4jLong> shapeA;
            std::vector<Nd4jLong> shapeB;
            auto ctx = new Context(1);

            if(rank == 3){
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {32, 1024, 1024}), true);
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {32, 1024, 1024}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {32, 1024, 1024}), true);
            } else {
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {4, 8, 1024, 1024}), true);
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {4, 8, 1024, 1024}), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {4, 8, 1024, 1024}), true);
            }

            return ctx;
        };

        nd4j::ops::matmul mmul;
        DeclarableBenchmark benchmark(mmul, "mmul (batch)");
        output += helper.runOperationSuit(&benchmark, generator, b, "MMul (batch)");

        return output;
    }

    static std::string gemmRegularBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        for (int o = 0; o <= 1; o++) {
            char resultOrder = (o == 0 ? 'f' : 'c');
            for (int tA = 0; tA <= 1; tA++) {
                for (int tB = 0; tB <= 1; tB++) {


                    IntPowerParameters pa("sz", 2, 7, gemmRegularUpperPow, 2);          //2^7=128, 2^9=512, 2^11=2048

                    ParametersBatch b({&pa});

                    auto generator = PARAMETRIC_XYZ() {
                        auto s = p.getIntParam("sz");
                        auto A = NDArrayFactory::create_<float>('c', {s, s});
                        auto B = NDArrayFactory::create_<float>('c', {s, s});
                        auto C = NDArrayFactory::create_<float>(resultOrder, {s, s});

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

                    output += helper.runOperationSuit(&mb, generator, b, n.c_str());
                }
            }
        }

        return output;
    }

    static std::string scatterOpBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 10, gatherOpPowLimit, 4);      //2^10 to 2^26 in steps of 4
        ParametersBatch batch({&length});

        //Gather 1D tests - 1d ref, 1d indices, 1d updates -> 1d output
        nd4j::ops::scatter_upd scatter_update1;
        DeclarableBenchmark sa1d(scatter_update1, "scatter_update1d");
        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int length = p.getIntParam("length");
            auto in = NDArrayFactory::create_<float>('c', {length});
            auto indices = NDArrayFactory::create_<int>('c', {length});
            auto updates = NDArrayFactory::create_<float>('c', {length});

            int* a = new int[length];
            for( int i=0; i<length; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + length-1));
            for( int i=0; i<length; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setInputArray(2, updates, true);
            ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
            ctx->markInplace(true);
            return ctx;
        };

        output += helper.runOperationSuit(&sa1d, generator, batch, "Scatter Update - 1d");

        //Gather 2D tests - 2d input, 1d indices, 2d updates -> 2d output
        IntPowerParameters rows("rows", 2, 8, gatherOpPowLimit2, 4);      //2^10 to 2^16 in steps of 2: 2^10, ..., 2^20
        PredefinedParameters cols("cols", {32});
        ParametersBatch batch2({&rows, &cols});
        nd4j::ops::scatter_upd scatter_update2;
        DeclarableBenchmark sa2d(scatter_update2, "scatter_update2d");
        auto generator2 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int rows = p.getIntParam("rows");
            int cols = p.getIntParam("cols");
            auto in = NDArrayFactory::create_<float>('c', {rows, cols});
            auto indices = NDArrayFactory::create_<int>('c', {rows});
            auto updates = NDArrayFactory::create_<float>('c', {rows, cols});

            int* a = new int[rows];
            for( int i=0; i<rows; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + rows-1));
            for( int i=0; i<rows; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setInputArray(2, updates, true);
            ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
            ctx->markInplace(true);
            return ctx;
        };

        output += helper.runOperationSuit(&sa2d, generator2, batch2, "Scatter Update - 2d");

        //Gather 3D tests - 3d input, 1d indices -> 3d output
        IntPowerParameters sz0("sz0", 2, 8, gatherOpPowLimit3, 4);
        PredefinedParameters sz1("sz1", {32});
        ParametersBatch batch3({&sz0, &sz1});
        nd4j::ops::scatter_upd scatter_update3;
        DeclarableBenchmark sa3d(scatter_update3, "scatter3d");
        auto generator3 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int sz0 = p.getIntParam("sz0");
            int sz1 = p.getIntParam("sz1");
            auto in = NDArrayFactory::create_<float>('c', {sz0, sz1, 512/sz1});
            auto indices = NDArrayFactory::create_<int>('c', {sz0});
            auto updates = NDArrayFactory::create_<float>('c', {sz0, sz1, 512/sz1});

            int* a = new int[sz0];
            for( int i=0; i<sz0; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + sz0-1));
            for( int i=0; i<sz0; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setInputArray(2, updates, true);
            ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
            ctx->markInplace(true);
            return ctx;
        };

        output += helper.runOperationSuit(&sa3d, generator3, batch3, "Scatter Update - 3d");
        return output;
    }

    static std::string gatherOpBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 10, gatherOpPowLimit, 4);      //2^10 to 2^22 in steps of 4
        ParametersBatch batch({&length});

        //Gather 1D tests - 1d input, 1d indices -> 1d output
        nd4j::ops::gather gather1;
        DeclarableBenchmark gather1d(gather1, "gather1d");
        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int length = p.getIntParam("length");
            auto in = NDArrayFactory::create_<float>('c', {length});
            auto indices = NDArrayFactory::create_<int>('c', {length});
            int* a = new int[length];
            for( int i=0; i<length; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + length-1));
            for( int i=0; i<length; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {length}), true);
            return ctx;
        };

        output += helper.runOperationSuit(&gather1d, generator, batch, "Gather - 1d");

        //Gather 2D tests - 2d input, 1d indices -> 2d output
        IntPowerParameters rows("rows", 2, 8, gatherOpPowLimit2, 4);      //2^10 to 2^20 in steps of 2: 2^10, ..., 2^20
        PredefinedParameters cols("cols", {32});
        ParametersBatch batch2({&rows, &cols});
        nd4j::ops::gather gather2;
        DeclarableBenchmark gather2d(gather2, "gather2d");
        auto generator2 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int rows = p.getIntParam("rows");
            int cols = p.getIntParam("cols");
            auto in = NDArrayFactory::create_<float>('c', {rows, cols});
            auto indices = NDArrayFactory::create_<int>('c', {rows});

            int* a = new int[rows];
            for( int i=0; i<rows; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + rows-1));
            for( int i=0; i<rows; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {rows, cols}), true);
            return ctx;
        };

        output += helper.runOperationSuit(&gather2d, generator2, batch2, "Gather - 2d");

        //Gather 3D tests - 3d input, 1d indices -> 3d output
        IntPowerParameters sz0("sz0", 2, 8, gatherOpPowLimit3, 4);      //2^8 to 2^16 in steps of 4
        PredefinedParameters sz1("sz1", {32});
        ParametersBatch batch3({&sz0, &sz1});
        nd4j::ops::gather gather3;
        DeclarableBenchmark gather3d(gather3, "gather3d");
        auto generator3 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int sz0 = p.getIntParam("sz0");
            int sz1 = p.getIntParam("sz1");
            auto in = NDArrayFactory::create_<float>('c', {sz0, sz1, 512/sz1});
            auto indices = NDArrayFactory::create_<int>('c', {sz0});

            int* a = new int[sz0];
            for( int i=0; i<sz0; i++ ){
                a[i] = i;
            }
            srand(12345);
            std::random_shuffle(a, (a + sz0-1));
            for( int i=0; i<sz0; i++ ){
                indices->p(i, a[i]);
            }
            delete[] a;

            ctx->setInputArray(0, in, true);
            ctx->setInputArray(1, indices, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {sz0, sz1, 512/sz1}), true);
            return ctx;
        };

        output += helper.runOperationSuit(&gather3d, generator3, batch3, "Gather - 3d");

        return output;
    }

    static std::string mismatchedOrdersAssignBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters rows("rows", 2, 2, mismatchedAssignPowLimit, 4);      //2^2 to 2^26 in steps of 2 - 2^1=2, ..., 2^26=67108864
        BoolParameters cf("cf");

        ParametersBatch batch({&rows, &cf});

        auto generator = PARAMETRIC_XZ() {
            int numElements = 67108864;    //2^26
            int rows = p.getIntParam("rows");
            int cols = numElements / rows;
            bool c = p.getIntParam("cf");

            auto arr = NDArrayFactory::create_<float>(c ? 'c' : 'f', {rows, cols});
            auto arr2 = NDArrayFactory::create_<float>(c ? 'f' : 'c', {rows, cols});
            x.push_back(arr);
            z.push_back(arr2);
        };

        TransformBenchmark tb(transform::AnyOps::Assign, "assign");
        output += helper.runOperationSuit(&tb, generator, batch, "C->F and F->C Assign");

        //Also test: NCHW to NHWC and back
        BoolParameters nchw("nchw");
        ParametersBatch batch2({&nchw});
        auto generator2 = PARAMETRIC_XZ() {
            bool nchw = p.getIntParam("nchw");

            if(nchw) {
                auto orig = NDArrayFactory::create_<float>('c', {16, 32, 64, 64});
                orig->permutei({0,2,3,1});
                x.push_back(orig);
                z.push_back(NDArrayFactory::create_<float>('c', {16, 64, 64, 32}));
            } else {
                auto orig = NDArrayFactory::create_<float>('c', {16, 64, 64, 32});
                orig->permutei({0,3,1,2});
                x.push_back(orig);
                z.push_back(NDArrayFactory::create_<float>('c', {16, 32, 64, 64}));
            }
        };

        TransformBenchmark tb2(transform::AnyOps::Assign, "assign_nchw");
        output += helper.runOperationSuit(&tb2, generator2, batch2, "nchw->nhwc and nhwc->nchw Assign");
        return output;
    }

    static std::string broadcastOpsMatrixBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Broadcast ops: matrices for rank 3, 4, 5
        for( int rank=3; rank <= broadcastMatrixRankLimit; rank++ ){
            int numAxisTests = -1;
            if(rank == 3){
                numAxisTests = 3;
            } else if(rank == 4){
                numAxisTests = 6;
            } else if(rank == 5){
                numAxisTests = 10;
            }

            IntParameters testNum("testNum", 0,numAxisTests-1,1);
            ParametersBatch b({&testNum});

            auto generator = PARAMETRIC_D(){
                int n = p.getIntParam("testNum");
                std::vector<int> axis({});
                switch(n){
                    //rank 3+
                    case 0:
                        axis = std::vector<int>({0,1});
                        break;
                    case 1:
                        axis = std::vector<int>({0,2});
                        break;
                    case 2:
                        axis = std::vector<int>({1,2});
                        break;
                        //rank 4+
                    case 3:
                        axis = std::vector<int>({0,3});
                        break;
                    case 4:
                        axis = std::vector<int>({1,3});
                        break;
                    case 5:
                        axis = std::vector<int>({2,3});
                        break;
                        //Rank 5
                    case 6:
                        axis = std::vector<int>({0,4});
                        break;
                    case 7:
                        axis = std::vector<int>({1,4});
                        break;
                    case 8:
                        axis = std::vector<int>({2,4});
                        break;
                    case 9:
                        axis = std::vector<int>({3,4});
                        break;
                }


                std::vector<Nd4jLong> shape({});
                std::vector<Nd4jLong> toBcShape({});
                int vectorLength;
                if(rank == 3){
                    shape = std::vector<Nd4jLong>({64,64,64});
                    toBcShape = std::vector<Nd4jLong>({64,64,64});
                    vectorLength = 64;
                } else if(rank == 4){
                    shape = std::vector<Nd4jLong>({32,32,32,32});
                    toBcShape = std::vector<Nd4jLong>({32,32,32,32});
                    vectorLength = 32;
                } else if(rank == 5){
                    shape = std::vector<Nd4jLong>({16,16,16,16,16});
                    toBcShape = std::vector<Nd4jLong>({16,16,16,16,16});
                    vectorLength = 16;
                }

                for( int i=0; i<rank; i++ ){
                    if(axis[0] == i || axis[1] == i){
                        continue;
                    }
                    toBcShape[i] = 1;
                }

                auto ctx = new Context(1);
                ctx->setInputArray(0, NDArrayFactory::create_<float>('c', shape), true);
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', toBcShape), true);
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', shape), true);
                return ctx;
            };

            std::string name;
            name += "Broadcast Matrix Add (Custom) - Rank";
            name += std::to_string(rank);

            nd4j::ops::add op;
            DeclarableBenchmark benchmark(op, "add");
            output += helper.runOperationSuit(&benchmark, generator, b, name.c_str());
        }

        return output;
    }


    static std::string broadcast2dBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        PredefinedParameters rows("rows", {65536});
        IntPowerParameters cols("cols", 2, 2, limit10, 4);      //2^2, 2^6, 2^10
        BoolParameters axis("axis");
        BoolParameters inplace("inplace");

        ParametersBatch batch({&rows, &cols, &axis, &inplace});

        auto generator = PARAMETRIC_D() {
            auto a = p.getIntParam("axis");
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")});

            auto ctx = new Context(1);
            ctx->setInputArray(0, arr, true);
            if(a == 0){
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), 1}), true);
            } else {
                ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {1, p.getIntParam("cols")}), true);
            }
            if (p.getIntParam("inplace") == 1) {
                ctx->setOutputArray(0, arr);
                ctx->markInplace(true);
            } else {
                ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")}), true);
            }
            return ctx;
        };

        std::string s("add");
        nd4j::ops::add op;
        DeclarableBenchmark benchmark(op, "add");
        output += helper.runOperationSuit(&benchmark, generator, batch, "Broadcast (Custom) Add - 2d");
        return output;
    }

    static std::string broadcastBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        //Broadcast ops: vectors for rank 2, 3, 4, 5
        for( int axis=0; axis<=1; axis++ ){
            PredefinedParameters rows("rows", {65536});
            IntPowerParameters cols("cols", 2, 2, limit10, 4);      //2^1 to 2^10 in steps of 2 - 2^1=2, ..., 2^10=1024
            BoolParameters inplace("inplace");

            ParametersBatch batch({&rows, &cols, &inplace});

            auto generator = PARAMETRIC_XYZ() {
                auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")});
                x.push_back(arr);
                if(axis == 0){
                    y.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("rows")}));
                } else {
                    y.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("cols")}));
                }
                if (p.getIntParam("inplace") == 1) {
                    z.push_back(arr);
                } else {
                    z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")}));
                }
            };

            std::string s("bAdd"); s += std::to_string(axis); s += "r2";
            BroadcastBenchmark bAdd(broadcast::Add, s, {axis});
            output += helper.runOperationSuit(&bAdd, generator, batch, "Broadcast Add - Rank 2");
        }

        for( int rank=3; rank<=5; rank++ ){
            for( int axis=1; axis<rank; axis++ ){
                std::vector<Nd4jLong> shape({});
                int vectorLength;
                if(rank == 3){
                    shape = std::vector<Nd4jLong>({32,128,128});
                    vectorLength = 128;
                } else if(rank == 4){
                    shape = std::vector<Nd4jLong>({16,64,64,64});
                    vectorLength = 64;
                } else if(rank == 5){
                    shape = std::vector<Nd4jLong>({16,48,48,48,48});
                    vectorLength = 48;
                }

                ParametersBatch batch({});

                //Note: always inplace here
                auto generator = PARAMETRIC_XYZ() {
                    auto arr = NDArrayFactory::create_<float>('c', shape);
                    x.push_back(arr);
                    y.push_back(NDArrayFactory::create_<float>('c', {vectorLength}));
                    z.push_back(arr);
                };

                std::string name("bArr-r"); name += std::to_string(rank); name += "a"; name += std::to_string(axis);
                BroadcastBenchmark bAdd(broadcast::Add, name, {axis});
                std::string n2("Broadcast Add - Rank"); n2 += std::to_string(rank); n2 += " - axis="; n2 += std::to_string(axis);
                output += helper.runOperationSuit(&bAdd, generator, batch, n2.c_str());
            }
        }

        return output;
    }

    static std::string fastStridedReductionNonEws() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters stride("stride", 2, 0, 10, 2);          //2^0=1, ..., 2^10=1024

        ParametersBatch batch({&stride});

        //This is an edge case: technically an EWS *should* be available here
        auto generator1 = PARAMETRIC_XYZ() {
            auto stride = p.getIntParam("stride");
            auto arr = NDArrayFactory::create_<float>('c', {131072 + (stride == 1 ? 0 : 1), stride});

            NDArray* strided;
            if(stride == 1){
                strided = arr;
            } else {
                IndicesList indices({NDIndex::interval(0,131072), NDIndex::interval(0,1)});
                strided = arr->subarray(indices);        //All rows, first column
                delete arr;
            }

            strided->assign(1.0);
            x.push_back(strided);
            y.push_back(nullptr);
            z.push_back(NDArrayFactory::create_<float>(0.0f));
        };

        ReductionBenchmark rbSum(reduce::SameOps::Sum, "stridedSum");
        output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator1), batch, "Strided Sum - No EWS Test 1");


        //No EWS defined for this case
        auto generator2 = PARAMETRIC_XYZ() {
            auto stride = p.getIntParam("stride");
            auto arr = NDArrayFactory::create_<float>('c', {(stride == 1 ? 1 : 2) * 1024, 1024, stride});

            NDArray* strided;
            if(stride == 1){
                strided = arr;
            } else {
                IndicesList indices({NDIndex::interval(0,2*1024,2), NDIndex::all(), NDIndex::interval(0,1)});
                strided = arr->subarray(indices);
                delete arr;
            }

            strided->assign(1.0);
            x.push_back(strided);
            y.push_back(nullptr);
            z.push_back(NDArrayFactory::create_<float>(0.0f));
        };

        ReductionBenchmark rbSum2(reduce::SameOps::Sum, "stridedSumNoEWS");
        output += helper.runOperationSuit(&rbSum2, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator2), batch, "Strided Sum - No EWS Test 2");

        return output;
    }

    static std::string fastStridedReductionIrregular() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 12, stridedReductionPowLimit, 4);      //2^12 to 2^20 in steps of 4
        PredefinedParameters stride("stride", {26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                               122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                                               1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028});

        ParametersBatch batch({&length, &stride});

        auto generator = PARAMETRIC_XYZ() {
            auto stride = p.getIntParam("stride");
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

            NDArray* strided;
            if(stride == 1){
                strided = arr;
            } else {
                IndicesList indices({NDIndex::all(), NDIndex::interval(0,1)});
                strided = arr->subarray(indices);        //All rows, first column
                delete arr;
            }

            strided->assign(1.0);
            x.push_back(strided);
            y.push_back(nullptr);
            z.push_back(NDArrayFactory::create_<float>(0.0f));
        };

        ReductionBenchmark rbSum(reduce::SameOps::Sum, "stridedSum");

        output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Sum - Irregular Strides");

        return output;
    }

    static std::string fastStridedReductionsRegular() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 12, stridedReductionPowLimit, 4);      //2^12 to 2^20 in steps of 4
        IntPowerParameters stride("stride", 2, 0, 10);          //2^0=1, ..., 2^10=1024

        ParametersBatch batch({&length, &stride});

        auto generator = PARAMETRIC_XYZ() {
            auto stride = p.getIntParam("stride");
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

            NDArray* strided;
            if(stride == 1){
                strided = arr;
            } else {
                IndicesList indices({NDIndex::all(), NDIndex::point(0)});
                strided = arr->subarray(indices);        //All rows, first column
                delete arr;
            }

            strided->assign(1.0);
            x.push_back(strided);
            y.push_back(nullptr);
//            z.push_back(NDArrayFactory::create_<float>(0.0f));
            z.push_back(NDArrayFactory::create_<float>('c', {1}));
        };

        ReductionBenchmark rbSum(reduce::SameOps::Sum, "Strided Sum");

        output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Sum - Regular Strides (powers of 2)");

        auto generator3 = PARAMETRIC_D(){
            auto ctx = new Context(1);
            auto stride = p.getIntParam("stride");
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

            NDArray* strided;
            if(stride == 1){
                strided = arr;
            } else {
                IndicesList indices({NDIndex::all(), NDIndex::point(0)});
                strided = arr->subarray(indices);        //All rows, first column
                delete arr;
            }

            strided->assign(1.0);
            ctx->setInputArray(0, strided, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong>('c', {1}), true);
            auto iargs = new Nd4jLong[1];
            iargs[0] = 0;
            ctx->setIArguments(iargs, 1);
            delete[] iargs;
            return ctx;
        };

        nd4j::ops::argmax opArgmax;
        DeclarableBenchmark dbArgmax(opArgmax, "stridedArgmax");
        output += helper.runOperationSuit(&dbArgmax, generator3, batch, "Strided Argmax");
        return output;
    }

    static std::string fastReduceAlongDimBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        int length[] = {1024*1024, 64*1024*1024};
        int powLimit[] = {10, 20, 26};
        int powStep[] = {2, 2, 4};

        for( int i=0; i < limit3; i++ ){
            IntPowerParameters rows("rows", 2, 0, powLimit[i], powStep[i]);
            BoolParameters dim("dim");


            ParametersBatch batch({&rows, &dim});

            auto generator = PARAMETRIC_XYZ() {
                int rows = p.getIntParam("rows");
                int cols = length[i] / rows;
                int dim = p.getIntParam("dim");
                auto arr = NDArrayFactory::create_<float>('c', {rows, cols});


                x.push_back(arr);
                y.push_back(NDArrayFactory::create_<Nd4jLong>(dim));

                NDArray* result;
                if(dim == 0){
                    result = NDArrayFactory::create_<float>('c', {cols});
                } else {
                    result = NDArrayFactory::create_<float>('c', {rows});
                }
                z.push_back(result);
            };

            ReductionBenchmark rbSum(reduce::SameOps::Sum, "sum");
            ReductionBenchmark rbMax(reduce::SameOps::Max, "max");

            std::string s1("Sum Along Dimension - ");
            s1 += std::to_string(length[i]);

            output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s1.c_str());


            auto generator3 = PARAMETRIC_D(){
                auto ctx = new Context(1);
                int rows = p.getIntParam("rows");
                int cols = length[i] / rows;
                int dim = p.getIntParam("dim");
                auto arr = NDArrayFactory::create_<float>('c', {rows, cols});

                Nd4jLong* dimArg = new Nd4jLong[1];
                dimArg[0] = dim;
                ctx->setIArguments(dimArg, 1);
                delete[] dimArg;

                ctx->setInputArray(0, arr, true);

                NDArray* result;
                if(dim == 0){
                    result = NDArrayFactory::create_<Nd4jLong>('c', {cols});
                } else {
                    result = NDArrayFactory::create_<Nd4jLong>('c', {rows});
                }
                ctx->setOutputArray(0, result, true);
                return ctx;
            };

            std::string s5("Argmax Along Dimension - ");
            s5 += std::to_string(length[i]);

            nd4j::ops::argmax opArgmax;
            DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
            output += helper.runOperationSuit(&dbArgmax, generator3, batch, s5.c_str());
        }

        return output;
    }

    static std::string fastReduceToScalarBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 10, reduceScalarPowLimit, 4);      //2^10 to 2^26 in steps of 4

        ParametersBatch batch({&length});

        auto generator = PARAMETRIC_XYZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});

            x.push_back(arr);
            y.push_back(nullptr);
            z.push_back(NDArrayFactory::create_<float>(0.0f));
        };

        ReductionBenchmark rbSum(reduce::SameOps::Sum, "sum");

        output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Sum - Full Array Reduction");

        //Index reduction
        nd4j::ops::argmax opArgmax;
        DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
        auto generator3 = PARAMETRIC_D(){
            auto ctx = new Context(1);

            ctx->setInputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("length")}), true);
            ctx->setInputArray(1, NDArrayFactory::create_<Nd4jLong>((Nd4jLong)0), true);
            ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong>(0), true);

            return ctx;
        };
        output += helper.runOperationSuit(&dbArgmax, generator3, batch, "Argmax Full Array Reduction");

        return output;
    }

    static std::string fastNonEwsTransformBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);
        IntPowerParameters rowcol("rowcol", 2, 2, nonEwsPowLimit, 4);      //2^2 to 2^14 in steps of 4 -> non-inplace case: 2x 2^10 x 2^10 = 128mb
        BoolParameters inplace("inplace");

        ParametersBatch batch({&rowcol, &inplace});

        auto generator = PARAMETRIC_XZ() {
            int r = p.getIntParam("rowcol");
            auto arr = NDArrayFactory::create_<float>('c', {r, r+1});
            IndicesList indices({NDIndex::all(), NDIndex::interval(0,r-1)});
            auto view = arr->subarray(indices);
            //nd4j_printf("VIEW ARRAY: rows=%lld, columns=%lld", view->sizeAt(0), view->sizeAt(1));
            x.push_back(view);
            if(p.getIntParam("inplace") == 1){
                z.push_back(view);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {r,r}));
            }
            delete arr;
        };

        ScalarBenchmark sbLRelu(scalar::Ops::LeakyRELU, "LeakyRELU_View");
        sbLRelu.setY(NDArrayFactory::create_<float>(0.0));

        TransformBenchmark tbExp(transform::StrictOps::Exp, "exp view");

        output += helper.runOperationSuit(&sbLRelu, generator, batch, "LeakyRELU View");
        output += helper.runOperationSuit(&tbExp, generator, batch, "Exp View");

        return output;
    }

    static std::string fastPairwiseBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);
        IntPowerParameters length("length", 2, 10, pairwisePowLimit, 4);      //2^10 to 2^26 in steps of 4 -> max is 512mb
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XYZ() {
            auto arr1 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            auto arr2 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            x.push_back(arr1);
            y.push_back(arr2);
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr1);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
            }
        };

        PairwiseBenchmark pb1(pairwise::Ops::Add, "Add");
        output += helper.runOperationSuit(&pb1, generator, batch, "Pairwise Add");

        PairwiseBenchmark pb2(pairwise::Ops::Add, "Multiply");
        output += helper.runOperationSuit(&pb2, generator, batch, "Pairwise Multiply");

        return output;
    }

    static std::string heavyTransformsBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);
        IntPowerParameters length("length", 2, 10, heavyPowLimit, 4);      //2^10 to 2^22, steps of 4
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            arr->assign(1.0);
            x.push_back(arr);
            if (p.getIntParam("inplace") == 1) {
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
            }
        };

        //Ops to test: erf (transform), betainc (custom), polygamma, synthetic ops?
        TransformBenchmark erf(transform::StrictOps::Erf, "Erf");
        output += helper.runOperationSuit(&erf, generator, batch, "Error Function (Erf)");

        ParametersBatch batch2({&length});
        nd4j::ops::polygamma op1;
        DeclarableBenchmark pg(op1, "polygamma");
        auto generator2 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            auto in0 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            in0->assign(0.25);
            auto in1 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            in1->assign(0.5);
            ctx->setInputArray(0, in0, true);
            ctx->setInputArray(1, in1, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("length")}), true);
            return ctx;
        };


        IntPowerParameters lengthBetaInc("length", 2, 10, heavyPowLimit, 4);      //2^10 to 2^22 in steps of 4
        ParametersBatch batch3({&lengthBetaInc});
        nd4j::ops::betainc op2;
        DeclarableBenchmark binc(op2, "betainc");
        auto generator3 = PARAMETRIC_D() {
            auto ctx = new Context(1);
            auto in0 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            in0->assign(0.25);
            auto in1 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            in1->assign(0.5);
            auto in2 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            in2->assign(0.75);
            ctx->setInputArray(0, in0, true);
            ctx->setInputArray(1, in1, true);
            ctx->setInputArray(2, in2, true);
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("length")}), true);
            return ctx;
        };

        output += helper.runOperationSuit(&pg, generator2, batch2, "PolyGamma Function");
        output += helper.runOperationSuit(&binc, generator3, batch3, "Incomplete Beta Function (BetaInc)");

        return output;
    }

    static std::string intermediateTransformsBenchmark() {
        std::string output;

        //Non-inplace: 2x 2^26 elements FP32 -> 512MB
        BenchmarkHelper helper(wIterations, rIterations);
        IntPowerParameters length("length", 2, 10, intermediateTransformPowLimit, 4);      //2^20 to 2^22 in steps of 4
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            arr->assign(1.0);
            x.push_back(arr);
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
            }
        };

        TransformBenchmark tbTanh(transform::StrictOps::Tanh, "tanh");
        TransformBenchmark tbGelu(transform::StrictOps::GELU, "gelu");

        output += helper.runOperationSuit(&tbTanh, generator, batch, "Tanh");
        output += helper.runOperationSuit(&tbGelu, generator, batch, "gelu");


        //2x 1024 cols x 2^18 = 2GB
        IntPowerParameters rows("rows", 2, 10, intermediateTransformPowLimit2, 4);
        PredefinedParameters cols("cols", {4, 128, 1024});

        ParametersBatch batch2({&rows, &cols, &inplace});

        auto generator2 = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")});
            arr->assign(1.0);
            x.push_back(arr);
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")}));
            }
        };

        //TransformBenchmark tbSoftmax(transform::StrictOps::SoftMax, "softmax");

        //output += helper.runOperationSuit(&tbSoftmax, generator2, batch2, "Softmax");

        return output;
    }

    static std::string fastTransformsBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);
        IntPowerParameters length("length", 2, 10, transformBenchmarkPowLimit, 4);      //2^10 to 2^30 in steps of 4 - 2^10, 2^14, ..., 2^26
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            arr->assign(1.0);
            x.push_back(arr);
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
            }
        };

        ScalarBenchmark sbLRelu(scalar::Ops::LeakyRELU, "LeakyRELU");
        sbLRelu.setY(NDArrayFactory::create_<float>(0.0));

        TransformBenchmark tbAbs(transform::SameOps::Abs, "abs");
        TransformBenchmark tbExp(transform::StrictOps::Exp, "exp");

        output += helper.runOperationSuit(&sbLRelu, generator, batch, "LeakyRELU");
        output += helper.runOperationSuit(&tbAbs, generator, batch, "Abs");
        output += helper.runOperationSuit(&tbExp, generator, batch, "Exp");

        return output;
    }

    static std::string fastScalarBenchmark() {
        std::string output;
        BenchmarkHelper helper(wIterations, rIterations);

        IntPowerParameters length("length", 2, 10, scalarBenchmarkPowLimit, 4);      //2^10 to 2^30 in steps of 4 - 2^10, 2^14, ..., 2^26
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
            arr->assign(1.0);
            x.push_back(arr);
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
            }
        };

        ScalarBenchmark sbAdd(scalar::Ops::Add, "sAdd");
        ScalarBenchmark sbDiv(scalar::Ops::Divide, "sDiv");
        ScalarBenchmark sbPow(scalar::Ops::Pow, "sPow");


        sbAdd.setY(NDArrayFactory::create_<float>(3.14159265359));
        sbDiv.setY(NDArrayFactory::create_<float>(3.14159265359));
        sbPow.setY(NDArrayFactory::create_<float>(3.14159265359));


        output += helper.runOperationSuit(&sbAdd, generator, batch, "Scalar Addition - x.add(3.14159265359) - F32");
        output += helper.runOperationSuit(&sbDiv, generator, batch, "Scalar Division - x.div(3.14159265359) - F32");
        output += helper.runOperationSuit(&sbPow, generator, batch, "Scalar Power - x.pow(3.14159265359) - F32");

        return output;
    }


    static long nowMs(){
        auto s = std::chrono::system_clock::now().time_since_epoch();
        auto v = std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
        return v;
    }

    static long duration(long start){
        return nowMs() - start;
    }

    static long done(long start){
        long dur = duration(start);
        nd4j_printf("Done: %i ms\n", dur);
        return nowMs();
    }


    std::string FullBenchmarkSuit::runSuit() {
        std::string result;

        long start = nowMs();
        
        // set 1
        nd4j_printf("Running FullBenchmarkSuite.fastScalarBenchmark\n", "");
        result += fastScalarBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastTransformsBenchmark\n", "");
        result += fastTransformsBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.intermediateTransformsBenchmark\n", "");
        result += intermediateTransformsBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastPairwiseBenchmark\n", "");
        result += fastPairwiseBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.heavyTransformsBenchmark\n", "");
        result += heavyTransformsBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastNonEwsTransformBenchmark\n", "");
        result += fastNonEwsTransformBenchmark();
        start = done(start);

        // set 2
        nd4j_printf("Running FullBenchmarkSuite.fastReduceToScalarBenchmark\n", "");
        result += fastReduceToScalarBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastReduceAlongDimBenchmark\n", "");
        result += fastReduceAlongDimBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastStridedReductionsRegular\n", "");
        result += fastStridedReductionsRegular();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastStridedReductionIrregular\n", "");
        result += fastStridedReductionIrregular();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.fastStridedReductionNonEws\n", "");
        result += fastStridedReductionNonEws();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.broadcastBenchmark\n", "");
        result += broadcastBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.broadcast2dBenchmark\n", "");
        result += broadcast2dBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.broadcastOpsMatrixBenchmark\n", "");
        result += broadcastOpsMatrixBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.mismatchedOrdersAssignBenchmark\n", "");
        result += mismatchedOrdersAssignBenchmark();
        start = done(start);


        // set 3
        nd4j_printf("Running FullBenchmarkSuite.gatherOpBenchmark\n", "");
        result += gatherOpBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.scatterOpBenchmark\n", "");
        result += scatterOpBenchmark();
        start = done(start);

        // set 4
        nd4j_printf("Running FullBenchmarkSuite.gemmRegularBenchmark\n", "");
        result += gemmRegularBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.gemmIrregularBenchmark\n", "");
        result += gemmIrregularBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.rngBenchmark\n", "");
        result += rngBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.conv2dBenchmark\n", "");
        result += conv2dBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.pool2dBenchmark\n", "");
        result += pool2dBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.batchnormBenchmark\n", "");
        result += batchnormBenchmark();
        start = done(start);

        nd4j_printf("Running FullBenchmarkSuite.lstmBenchmark\n", "");
        result += lstmBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.conv3dBenchmark\n", "");
        result += conv3dBenchmark();
        start = done(start);
        nd4j_printf("Running FullBenchmarkSuite.maxPool3DBenchmark\n", "");
        result += maxPool3DBenchmark();
        start = done(start);
//        nd4j_printf("Running FullBenchmarkSuite.layerNormBenchmark\n", "");
//        result += layerNormBenchmark();
//        start = done(start);

        return result;
    }


}