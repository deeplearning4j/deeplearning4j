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

using namespace nd4j;
using namespace nd4j::graph;

class BenchmarkSet1 : public testing::Test {
public:

    BenchmarkSet1() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(BenchmarkSet1, FastScalar) {
    BenchmarkHelper helper(10, 1000);

    IntPowerParameters length("length", 2, 4, 30, 2);      //2^4 to 2^30 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^30
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
    ScalarBenchmark sbSub(scalar::Ops::Subtract, "sSub");
    ScalarBenchmark sbMul(scalar::Ops::Multiply, "sMul");
    ScalarBenchmark sbDiv(scalar::Ops::Divide, "sDiv");
//    ScalarBenchmark sbMax(scalar::Ops::Maximum, "sMax");
    ScalarBenchmark sbPow(scalar::Ops::Pow, "sPow");


    sbAdd.setY(NDArrayFactory::create_<float>(3.14159265359));
    sbSub.setY(NDArrayFactory::create_<float>(3.14159265359));
    sbMul.setY(NDArrayFactory::create_<float>(3.14159265359));
    sbDiv.setY(NDArrayFactory::create_<float>(3.14159265359));
//    sbMax.setY(NDArrayFactory::create_<float>(3.14159265359));
    sbPow.setY(NDArrayFactory::create_<float>(3.14159265359));


    helper.runOperationSuit(&sbAdd, generator, batch, "Scalar Addition - x.add(3.14159265359)");
    helper.runOperationSuit(&sbSub, generator, batch, "Scalar Subtraction - x.sub(3.14159265359)");
    helper.runOperationSuit(&sbMul, generator, batch, "Scalar Multiplication - x.mul(3.14159265359)");
    helper.runOperationSuit(&sbDiv, generator, batch, "Scalar Division - x.div(3.14159265359)");
//    helper.runOperationSuit(&sbMax, generator, batch, "Scalar Maximum - x.max(3.14159265359)");
    helper.runOperationSuit(&sbPow, generator, batch, "Scalar Power - x.pow(3.14159265359)");
}


TEST_F(BenchmarkSet1, FastTransforms) {
    BenchmarkHelper helper(10,1000);
    IntPowerParameters length("length", 2, 4, 30, 2);      //2^4 to 2^30 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^30
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

    ScalarBenchmark sbRelu(scalar::Ops::RELU, "RELU");
    sbRelu.setY(NDArrayFactory::create_<float>(0.0));
    ScalarBenchmark sbLRelu(scalar::Ops::LeakyRELU, "LeakyRELU");
    sbLRelu.setY(NDArrayFactory::create_<float>(0.0));

    TransformBenchmark tbAbs(transform::SameOps::Abs, "abs");
    TransformBenchmark tbSign(transform::SameOps::Sign, "sign");
    TransformBenchmark tbNeg(transform::SameOps::Neg, "neg");
    TransformBenchmark tbExp(transform::StrictOps::Exp, "exp");

    helper.runOperationSuit(&sbRelu, generator, batch, "RELU");
    helper.runOperationSuit(&sbLRelu, generator, batch, "LeakyRELU");
    helper.runOperationSuit(&tbAbs, generator, batch, "Abs");
    helper.runOperationSuit(&tbSign, generator, batch, "Sign");
    helper.runOperationSuit(&tbNeg, generator, batch, "Neg");
    helper.runOperationSuit(&tbExp, generator, batch, "Exp");
}


TEST_F(BenchmarkSet1, IntermediateTransforms) {
    BenchmarkHelper helper;
    IntPowerParameters length("length", 2, 4, 30, 2);      //2^4 to 2^30 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^30
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
    TransformBenchmark tbCosh(transform::StrictOps::Cosh, "cosh");
    TransformBenchmark tbSinh(transform::StrictOps::Sinh, "sinh");
    TransformBenchmark tbAcosh(transform::StrictOps::ACosh, "acosh");
    TransformBenchmark tbCosine(transform::StrictOps::Cosine, "cosine");
    TransformBenchmark tbSigmoid(transform::StrictOps::Sigmoid, "sigmoid");
    TransformBenchmark tbGelu(transform::StrictOps::GELU, "gelu");

    helper.runOperationSuit(&tbTanh, generator, batch, "Tanh");
    helper.runOperationSuit(&tbCosh, generator, batch, "Cosh");
    helper.runOperationSuit(&tbSinh, generator, batch, "Sinh");
    helper.runOperationSuit(&tbAcosh, generator, batch, "Acosh");
    helper.runOperationSuit(&tbSigmoid, generator, batch, "Sigmoid");
    helper.runOperationSuit(&tbGelu, generator, batch, "gelu");



    IntPowerParameters rows("rows", 2, 4, 20, 2);      //2^4 to 2^20 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^20
    PredefinedParameters cols("cols", {4, 16, 128, 512, 1024});

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

    TransformBenchmark tbSoftmax(transform::StrictOps::SoftMax, "softmax");

    helper.runOperationSuit(&tbSoftmax, generator2, batch2, "Softmax");
}

#define PARAMETRIC_D() [&] (Parameters &p) -> Context*

TEST_F(BenchmarkSet1, HeavyTransforms) {
    BenchmarkHelper helper;
    IntPowerParameters length("length", 2, 4, 30, 2);      //2^4 to 2^30 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^30
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
    helper.runOperationSuit(&erf, generator, batch, "Error Function (Erf)");

    ParametersBatch batch2({&length});
    nd4j::ops::polygamma op1;
    DeclarableBenchmark pg(op1, "polygamma");
    auto generator2 = PARAMETRIC_D() {
        auto ctx = new Context(1);
        auto in0 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
        in0->assign(0.25);
        auto in1 = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});
        in1->assign(0.5);
        ctx->setInputArray(0, in0);
        ctx->setInputArray(1, in1);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
        return ctx;
    };


    IntPowerParameters lengthBetaInc("length", 2, 4, 24, 2);      //2^4 to 2^24 in steps of 2 - NOTE: Betainc is quite slow
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
        ctx->setInputArray(0, in0);
        ctx->setInputArray(1, in1);
        ctx->setInputArray(2, in2);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("length")}));
        return ctx;
    };

    helper.runOperationSuit(&pg, generator2, batch2, "PolyGamma Function");
    helper.runOperationSuit(&binc, generator3, batch3, "Incomplete Beta Function (BetaInc)");
}


TEST_F(BenchmarkSet1, Pairwise) {
    BenchmarkHelper helper(10,1000);
    IntPowerParameters length("length", 2, 4, 30, 2);      //2^4 to 2^30 in steps of 2 - 2^4, 2^6, 2^8, ..., 2^30
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
    helper.runOperationSuit(&pb1, generator, batch, "Pairwise Add");

    PairwiseBenchmark pb2(pairwise::Ops::Add, "Multiply");
    helper.runOperationSuit(&pb2, generator, batch, "Pairwise Multiply");
}

TEST_F(BenchmarkSet1, TransformViewNoEws) {
    BenchmarkHelper helper(10,100);
    IntPowerParameters rowcol("rowcol", 2, 1, 15, 2);      //2^1 to 2^15 in steps of 2
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
    };

    ScalarBenchmark sbRelu(scalar::Ops::RELU, "RELU_View");
    sbRelu.setY(NDArrayFactory::create_<float>(0.0));
    ScalarBenchmark sbLRelu(scalar::Ops::LeakyRELU, "LeakyRELU_View");
    sbLRelu.setY(NDArrayFactory::create_<float>(0.0));

    TransformBenchmark tbAbs(transform::SameOps::Abs, "abs");
    TransformBenchmark tbExp(transform::StrictOps::Exp, "exp");

    helper.runOperationSuit(&sbRelu, generator, batch, "RELU View ");
    helper.runOperationSuit(&sbLRelu, generator, batch, "LeakyRELU View");
    helper.runOperationSuit(&tbAbs, generator, batch, "Abs View");
    helper.runOperationSuit(&tbExp, generator, batch, "Exp View");
}
