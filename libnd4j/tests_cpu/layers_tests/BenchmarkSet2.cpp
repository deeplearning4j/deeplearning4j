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

class BenchmarkSet2 : public testing::Test {
public:

    BenchmarkSet2() {
        printf("\n");
        fflush(stdout);
    }
};

#define PARAMETRIC_D() [&] (Parameters &p) -> Context*

TEST_F(BenchmarkSet2, FullArrayReductions) {
    BenchmarkHelper helper;

    IntPowerParameters length("length", 2, 4, 30, 2);      //2^3 to 2^30 in steps of 2

    ParametersBatch batch({&length});

    auto generator = PARAMETRIC_XYZ() {
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});

        x.push_back(arr);
        y.push_back(nullptr);
        z.push_back(NDArrayFactory::create_<float>(0.0f));
    };

    ReductionBenchmark rbSum(reduce::SameOps::Sum, "sum");
    ReductionBenchmark rbProd(reduce::SameOps::Prod, "prod");
    ReductionBenchmark rbMax(reduce::SameOps::Max, "max");

    helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Sum - Full Array Reduction");
    helper.runOperationSuit(&rbProd, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Product - Full Array Reduction");
    helper.runOperationSuit(&rbMax, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Maximum - Full Array Reduction");


    auto generator2 = PARAMETRIC_D(){
        auto ctx = new Context(1);
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});

        ctx->setInputArray(0, arr);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>(0.0f));
        return ctx;
    };

    nd4j::ops::reduce_logsumexp opLSE;
    DeclarableBenchmark dbLSE(opLSE, "LogSumExp");
    //helper.runOperationSuit(&dbLSE, generator2, batch, "LogSumExp");

    //Index reduction
    nd4j::ops::argmax opArgmax;
    DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
    auto generator3 = PARAMETRIC_D(){
        auto ctx = new Context(1);
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length")});

        ctx->setInputArray(0, arr);
        ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong>(0));
        ctx->setInputArray(1, NDArrayFactory::create_<Nd4jLong>((Nd4jLong)0));

        return ctx;
    };
    helper.runOperationSuit(&dbArgmax, generator3, batch, "Argmax Full Array Reduction");
}

TEST_F(BenchmarkSet2, ReductionAlongDim) {
    BenchmarkHelper helper;

    int length[] = {1024, 1024*1024, 1024*1024*1024};
    int pow[] = {10, 20, 30};

    for( int i=0; i<3; i++ ){
        IntPowerParameters rows("rows", 2, 0, pow[i], 1);
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
        ReductionBenchmark rbProd(reduce::SameOps::Prod, "prod");
        ReductionBenchmark rbMax(reduce::SameOps::Max, "max");

        std::string s1("Sum Along Dimension - ");
        s1 += std::to_string(length[i]);
        std::string s2("Product Along Dimension - ");
        s2 += std::to_string(length[i]);
        std::string s3("Maximum Along Dimension - ");
        s3 += std::to_string(length[i]);

        helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s1.c_str());
        helper.runOperationSuit(&rbProd, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s2.c_str());
        helper.runOperationSuit(&rbMax, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s3.c_str());


        auto generator2 = PARAMETRIC_D(){
            auto ctx = new Context(1);
            int rows = p.getIntParam("rows");
            int cols = length[i] / rows;
            int dim = p.getIntParam("dim");
            auto arr = NDArrayFactory::create_<float>('c', {rows, cols});

            Nd4jLong* dimArg = new Nd4jLong[1];
            dimArg[0] = dim;
            ctx->setIArguments(dimArg, 1);
            delete[] dimArg;

            ctx->setInputArray(0, arr);

            NDArray* result;
            if(dim == 0){
                result = NDArrayFactory::create_<float>('c', {cols});
            } else {
                result = NDArrayFactory::create_<float>('c', {rows});
            }
            ctx->setOutputArray(0, result);
            return ctx;
        };

        std::string s4("LogSumExp - ");
        s4 += std::to_string(length[i]);

        nd4j::ops::reduce_logsumexp opLSE;
        DeclarableBenchmark dbLSE(opLSE, "LogSumExp");
        //helper.runOperationSuit(&dbLSE, generator2, batch, s4.c_str());


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

            ctx->setInputArray(0, arr);

            NDArray* result;
            if(dim == 0){
                result = NDArrayFactory::create_<Nd4jLong>('c', {cols});
            } else {
                result = NDArrayFactory::create_<Nd4jLong>('c', {rows});
            }
            ctx->setOutputArray(0, result);
            return ctx;
        };

        std::string s5("Argmax Along Dimension - ");
        s5 += std::to_string(length[i]);

        nd4j::ops::argmax opArgmax;
        DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
        helper.runOperationSuit(&dbArgmax, generator3, batch, s5.c_str());
    }


}


TEST_F(BenchmarkSet2, StridedReductionsRegular) {
    BenchmarkHelper helper;

    IntPowerParameters length("length", 2, 6, 20, 2);      //2^6 to 2^20 in steps of 2 - 2^6=64, ..., 2^20=1048576
    IntPowerParameters stride("stride", 2, 0, 10);          //2^0=1, ..., 2^10=1024

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
        }

        strided->assign(1.0);
        x.push_back(strided);
        y.push_back(nullptr);
        z.push_back(NDArrayFactory::create_<float>(0.0f));
    };

    ReductionBenchmark rbSum(reduce::SameOps::Sum, "stridedSum");
    ReductionBenchmark rbProd(reduce::SameOps::Prod, "stridedProd");

    helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Sum - Regular Strides (powers of 2)");
    helper.runOperationSuit(&rbProd, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Product - Regular Strides (powers of 2)");


    auto generator2 = PARAMETRIC_D(){
        auto ctx = new Context(1);
        auto stride = p.getIntParam("stride");
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

        NDArray* strided;
        if(stride == 1){
            strided = arr;
        } else {
            IndicesList indices({NDIndex::all(), NDIndex::interval(0,1)});
            strided = arr->subarray(indices);        //All rows, first column
        }

        strided->assign(1.0);
        ctx->setInputArray(0, strided);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>(0.0f));
        Nd4jLong* iargs = new Nd4jLong[1];
        iargs[0] = 0;
        ctx->setIArguments(iargs, 1);
        delete[] iargs;
        return ctx;
    };

    nd4j::ops::reduce_logsumexp opLSE;
    DeclarableBenchmark dbLSE(opLSE, "stridedLSE");
    //helper.runOperationSuit(&dbLSE, generator2, batch, "LogSumExp");

    auto generator3 = PARAMETRIC_D(){
        auto ctx = new Context(1);
        auto stride = p.getIntParam("stride");
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

        NDArray* strided;
        if(stride == 1){
            strided = arr;
        } else {
            IndicesList indices({NDIndex::all(), NDIndex::interval(0,1)});
            strided = arr->subarray(indices);        //All rows, first column
        }

        strided->assign(1.0);
        ctx->setInputArray(0, strided);
        ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong >(0));
        Nd4jLong* iargs = new Nd4jLong[1];
        iargs[0] = 0;
        ctx->setIArguments(iargs, 1);
        delete[] iargs;
        return ctx;
    };

    nd4j::ops::argmax opArgmax;
    DeclarableBenchmark dbArgmax(opArgmax, "stridedArgmax");
    helper.runOperationSuit(&dbArgmax, generator3, batch, "Argmax");
}

TEST_F(BenchmarkSet2, StridedReductionsIrregular) {
    BenchmarkHelper helper;

    IntPowerParameters length("length", 2, 6, 20, 2);      //2^6 to 2^20 in steps of 2 - 2^7=128, ..., 2^20=1048576
    PredefinedParameters stride("stride", {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                           120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                                           1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028});

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
        }

        strided->assign(1.0);
        x.push_back(strided);
        y.push_back(nullptr);
        z.push_back(NDArrayFactory::create_<float>(0.0f));
    };

    ReductionBenchmark rbSum(reduce::SameOps::Sum, "stridedSum");
    ReductionBenchmark rbProd(reduce::SameOps::Prod, "stridedProd");

    helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Sum - Irregular Strides");
    helper.runOperationSuit(&rbProd, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Strided Product - Irregular Strides");

    auto generator2 = PARAMETRIC_D(){
        auto ctx = new Context(1);
        auto stride = p.getIntParam("stride");
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("length"), stride});

        NDArray* strided;
        if(stride == 1){
            strided = arr;
        } else {
            IndicesList indices({NDIndex::all(), NDIndex::interval(0,1)});
            strided = arr->subarray(indices);        //All rows, first column
        }

        strided->assign(1.0);
        ctx->setInputArray(0, strided);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>(0.0f));
        Nd4jLong* iargs = new Nd4jLong[1];
        iargs[0] = 0;
        ctx->setIArguments(iargs, 1);
        delete[] iargs;
        return ctx;
    };

    nd4j::ops::reduce_logsumexp opLSE;
    DeclarableBenchmark dbLSE(opLSE, "stridedLSE");
    //helper.runOperationSuit(&dbLSE, generator2, batch, "LogSumExp");
}

TEST_F(BenchmarkSet2, StridedReductionsNoEWS) {
    BenchmarkHelper helper;

    IntPowerParameters stride("stride", 2, 0, 10);          //2^0=1, ..., 2^10=1024

    ParametersBatch batch({&stride});

    //This is an edge case: technically an EWS *should* be available here
    auto generator1 = PARAMETRIC_XYZ() {
        auto stride = p.getIntParam("stride");
        auto arr = NDArrayFactory::create_<float>('c', {1048576 + (stride == 1 ? 0 : 1), stride});

        NDArray* strided;
        if(stride == 1){
            strided = arr;
        } else {
            IndicesList indices({NDIndex::interval(0,1048576), NDIndex::interval(0,1)});
            strided = arr->subarray(indices);        //All rows, first column
        }

        strided->assign(1.0);
        x.push_back(strided);
        y.push_back(nullptr);
        z.push_back(NDArrayFactory::create_<float>(0.0f));
    };

    ReductionBenchmark rbSum(reduce::SameOps::Sum, "stridedSum");
    helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator1), batch, "Strided Sum - No EWS Test 1");


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
        }

        strided->assign(1.0);
        x.push_back(strided);
        y.push_back(nullptr);
        z.push_back(NDArrayFactory::create_<float>(0.0f));
    };

    ReductionBenchmark rbSum2(reduce::SameOps::Sum, "stridedSumNoEWS");
    helper.runOperationSuit(&rbSum2, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator2), batch, "Strided Sum - No EWS Test 2");
}


TEST_F(BenchmarkSet2, BroadcastOps) {
    BenchmarkHelper helper;

    //Broadcast ops: vectors for rank 2, 3, 4, 5
    for( int axis=0; axis<=1; axis++ ){
        PredefinedParameters rows("rows", {1024, 1048576});
        IntPowerParameters cols("cols", 2, 2, 10, 2);      //2^1 to 2^10 in steps of 2 - 2^1=2, ..., 2^10=1024
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
        helper.runOperationSuit(&bAdd, generator, batch, "Broadcast Add - Rank 2");

        std::string s2("bMul"); s2 += std::to_string(axis); s2 += "r2";
        BroadcastBenchmark bMul(broadcast::Multiply, s2, {axis});
        helper.runOperationSuit(&bMul, generator, batch, "Broadcast Multiply - Rank 2");

        std::string s3("bPow"); s3 += std::to_string(axis); s3 += "r2";
        BroadcastBenchmark bPow(broadcast::Pow, s3, {axis});
        helper.runOperationSuit(&bPow, generator, batch, "Broadcast Power - Rank 2");
    }

    for( int rank=3; rank<=5; rank++ ){
        for( int axis=1; axis<rank; axis++ ){
            std::vector<Nd4jLong> shape({});
            int vectorLength;
            if(rank == 3){
                shape = std::vector<Nd4jLong>({32,128,128});
                vectorLength = 128;
            } else if(rank == 4){
                shape = std::vector<Nd4jLong>({32,128,128,128});
                vectorLength = 128;
            } else if(rank == 5){
                shape = std::vector<Nd4jLong>({32,64,64,64,64});
                vectorLength = 64;
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
            helper.runOperationSuit(&bAdd, generator, batch, n2.c_str());

            std::string n3("bPow-r"); n3 += std::to_string(rank); n3 += "a"; n3 += std::to_string(axis);
            BroadcastBenchmark bPow(broadcast::Pow, n3, {axis});
            std::string n4("Broadcast Power - Rank"); n4 += std::to_string(rank); n4 += " - axis="; n4 += std::to_string(axis);
            helper.runOperationSuit(&bPow, generator, batch, n4.c_str());
        }
    }

    //Broadcast ops: matrices for rank 3, 4, 5
    for( int rank=3; rank<=5; rank++ ){
        int numAxisTests;
        if(rank == 3){
            numAxisTests = 1;
        } else if(rank == 4){
            numAxisTests = 3;
        } else if(rank == 5){
            numAxisTests = 6;
        }

        for( int n=0; n<numAxisTests; n++ ){
            std::vector<int> axis({});
            switch(n){
                case 0:
                    axis = std::vector<int>({1,2});
                    break;
                case 1:
                    axis = std::vector<int>({1,3});
                    break;
                case 2:
                    axis = std::vector<int>({2,3});
                    break;
                case 3:
                    axis = std::vector<int>({1,4});
                    break;
                case 4:
                    axis = std::vector<int>({2,4});
                    break;
                case 5:
                    axis = std::vector<int>({3,4});
                    break;
            }

            std::vector<Nd4jLong> shape({});
            int vectorLength;
            if(rank == 3){
                shape = std::vector<Nd4jLong>({32,128});
                vectorLength = 128;
            } else if(rank == 4){
                shape = std::vector<Nd4jLong>({32,128,128,128});
                vectorLength = 128;
            } else if(rank == 5){
                shape = std::vector<Nd4jLong>({32,64,64,64,64});
                vectorLength = 64;
            }

            ParametersBatch batch({});

            //Note: always inplace here
            auto generator = PARAMETRIC_XYZ() {
                auto arr = NDArrayFactory::create_<float>('c', shape);
                x.push_back(arr);
                y.push_back(NDArrayFactory::create_<float>('c', {vectorLength,vectorLength}));
                z.push_back(arr);
            };

            std::string name("matrix_bAdd-r"); name += std::to_string(rank); name += "a"; name += std::to_string(axis[0]); name += ","; name += std::to_string(axis[1]);
            BroadcastBenchmark bAdd(broadcast::Add, name, axis);
            std::string n2("Matrix Broadcast Add - Rank"); n2 += std::to_string(rank); n2 += " - axis="; n2 += std::to_string(axis[0]); n2 += ","; n2 += std::to_string(axis[1]);
            helper.runOperationSuit(&bAdd, generator, batch, n2.c_str());

            std::string n3("matrix_bPow-r"); n3 += std::to_string(rank); n3 += "a"; n3 += std::to_string(axis[0]); n3 += ","; n3 += std::to_string(axis[1]);
            BroadcastBenchmark bPow(broadcast::Pow, n3, axis);
            std::string n4("Matrix Broadcast Power - Rank"); n4 += std::to_string(rank); n4 += "a"; n4 += std::to_string(axis[0]); n4 += ","; n4 += std::to_string(axis[1]);
            helper.runOperationSuit(&bPow, generator, batch, n4.c_str());
        }
    }
}

TEST_F(BenchmarkSet2, BroadcastOps2d) {
    BenchmarkHelper helper;

    PredefinedParameters rows("rows", {1024, 1048576});
    IntPowerParameters cols("cols", 2, 2, 10, 2);      //2^1 to 2^10 in steps of 2 - 2^1=2, ..., 2^10=1024
    BoolParameters axis("axis");
    BoolParameters inplace("inplace");

    ParametersBatch batch({&rows, &cols, &axis, &inplace});

    auto generator = PARAMETRIC_D() {
        auto a = p.getIntParam("axis");
        auto arr = NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")});

        auto ctx = new Context(1);
        ctx->setInputArray(0, arr);
        if(a == 0){
            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), 1}));
        } else {
            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', {1, p.getIntParam("cols")}));
        }
        if (p.getIntParam("inplace") == 1) {
            ctx->setOutputArray(0, arr);
            ctx->markInplace(true);
        } else {
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {p.getIntParam("rows"), p.getIntParam("cols")}));
        }
        return ctx;
    };

    std::string s("add");
    nd4j::ops::add op;
    DeclarableBenchmark benchmark(op, "add");
    helper.runOperationSuit(&benchmark, generator, batch, "Broadcast (Custom) Add - 2d");
}

TEST_F(BenchmarkSet2, BroadcastOpsMatrix) {
    BenchmarkHelper helper;

    //Broadcast ops: matrices for rank 3, 4, 5
    for( int rank=3; rank<=5; rank++ ){
        int numAxisTests = -1;
        if(rank == 3){
            numAxisTests = 3;
        } else if(rank == 4){
            numAxisTests = 6;
        } else if(rank == 5){
            numAxisTests = 10;
        }

        IntParameters testNum("testNum", 1,numAxisTests,1);
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
                shape = std::vector<Nd4jLong>({512,512,512});
                toBcShape = std::vector<Nd4jLong>({512,512,512});
                vectorLength = 128;
            } else if(rank == 4){
                shape = std::vector<Nd4jLong>({128,128,128,128});
                toBcShape = std::vector<Nd4jLong>({128,128,128,128});
                vectorLength = 128;
            } else if(rank == 5){
                shape = std::vector<Nd4jLong>({64,64,64,64,64});
                toBcShape = std::vector<Nd4jLong>({64,64,64,64,64});
                vectorLength = 64;
            }

            for( int i=0; i<rank; i++ ){
                if(axis[0] == i || axis[1] == i){
                    continue;
                }
                toBcShape[i] = 1;
            }

            auto ctx = new Context(1);
            ctx->setInputArray(0, NDArrayFactory::create_<float>('c', shape));
            ctx->setInputArray(1, NDArrayFactory::create_<float>('c', toBcShape));
            ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', shape));
            return ctx;
        };

        std::string name;
        name += "Broadcast Matrix Add (Custom) - Rank";
        name += std::to_string(rank);

        nd4j::ops::add op;
        DeclarableBenchmark benchmark(op, "add");
        helper.runOperationSuit(&benchmark, generator, b, name.c_str());
    }
}


TEST_F(BenchmarkSet2, MismatchedOrderAssign) {
    BenchmarkHelper helper;

    IntPowerParameters rows("rows", 2, 1, 26, 2);      //2^1 to 2^26 in steps of 2 - 2^1=2, ..., 2^26=67108864
    BoolParameters cf("cf");

    ParametersBatch batch({&rows, &cf});

    auto generator = PARAMETRIC_XZ() {
        int numElements = 268435456;    //2^28
        int rows = p.getIntParam("rows");
        int cols = numElements / rows;
        bool c = p.getIntParam("cf");

        auto arr = NDArrayFactory::create_<float>(c ? 'c' : 'f', {rows, cols});
        auto arr2 = NDArrayFactory::create_<float>(c ? 'f' : 'c', {rows, cols});
        x.push_back(arr);
        z.push_back(arr2);
    };

    TransformBenchmark tb(transform::AnyOps::Assign, "assign");
    helper.runOperationSuit(&tb, generator, batch, "C->F and F->C Assign");

    //Also test: NCHW to NHWC and back
    BoolParameters nchw("nchw");
    PredefinedParameters hw("hw", {62, 63, 64, 65, 66});
    ParametersBatch batch2({&nchw, &hw});
    auto generator2 = PARAMETRIC_XZ() {
        int hw = p.getIntParam("hw");
        bool nchw = p.getIntParam("nchw");

        if(nchw) {
            auto orig = NDArrayFactory::create_<float>('c', {32, 64, hw, hw});
            auto permute = orig->permute({0,2,3,1});
            x.push_back(permute);
            z.push_back(NDArrayFactory::create_<float>('c', {32, hw, hw, 64}));
        } else {
            auto orig = NDArrayFactory::create_<float>('c', {32, hw, hw, 64});
            auto permute = orig->permute({0,3,1,2});
            x.push_back(permute);
            z.push_back(NDArrayFactory::create_<float>('c', {32, 64, hw, hw}));
        }
    };

    TransformBenchmark tb2(transform::AnyOps::Assign, "assign_nchw");
    helper.runOperationSuit(&tb2, generator2, batch2, "nchw->nhwc and nhwc->nchw Assign");
}

