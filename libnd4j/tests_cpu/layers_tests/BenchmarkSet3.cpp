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

#include <random>
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

class BenchmarkSet3 : public testing::Test {
public:

    BenchmarkSet3() {
        printf("\n");
        fflush(stdout);
    }
};

#define PARAMETRIC_D() [&] (Parameters &p) -> Context*

TEST_F(BenchmarkSet3, GatherOps) {
    BenchmarkHelper helper;

    IntPowerParameters length("length", 2, 10, 30, 2);      //2^10 to 2^30 in steps of 2: 2^10, ..., 2^30
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {length}));
        return ctx;
    };

    helper.runOperationSuit(&gather1d, generator, batch, "Gather - 1d");

    //Gather 2D tests - 2d input, 1d indices -> 2d output
    IntPowerParameters rows("rows", 2, 10, 20, 2);      //2^10 to 2^20 in steps of 2: 2^10, ..., 2^20
    PredefinedParameters cols("cols", {32, 256, 1024});
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {rows, cols}));
        return ctx;
    };

    helper.runOperationSuit(&gather2d, generator2, batch2, "Gather - 2d");

    //Gather 3D tests - 3d input, 1d indices -> 3d output
    IntPowerParameters sz0("sz0", 2, 10, 20, 2);      //2^10 to 2^20 in steps of 2: 2^10, ..., 2^20
    PredefinedParameters sz1("sz1", {256, 4});
    ParametersBatch batch3({&sz0, &sz1});
    nd4j::ops::gather gather3;
    DeclarableBenchmark gather3d(gather3, "gather3d");
    auto generator3 = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int sz0 = p.getIntParam("sz0");
        int sz1 = p.getIntParam("sz1");
        auto in = NDArrayFactory::create_<float>('c', {sz0, sz1, 1024/sz1});
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setOutputArray(0, NDArrayFactory::create_<float>('c', {sz0, sz1, 1024/sz1}));
        return ctx;
    };

    helper.runOperationSuit(&gather3d, generator3, batch3, "Gather - 3d");

}

TEST_F(BenchmarkSet3, ScatterOps) {
    BenchmarkHelper helper;

    IntPowerParameters length("length", 2, 10, 22, 2);      //2^10 to 2^22 in steps of 2
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setInputArray(2, updates);
        ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
        ctx->markInplace(true);
        return ctx;
    };

    helper.runOperationSuit(&sa1d, generator, batch, "Scatter Update - 1d");

    //Gather 2D tests - 2d input, 1d indices, 2d updates -> 2d output
    IntPowerParameters rows("rows", 2, 10, 18, 2);      //2^10 to 2^20 in steps of 2: 2^10, ..., 2^20
    PredefinedParameters cols("cols", {32, 256, 512});   //, 1024});
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setInputArray(2, updates);
        ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
        ctx->markInplace(true);
        return ctx;
    };

    helper.runOperationSuit(&sa2d, generator2, batch2, "Scatter Update - 2d");

    //Gather 3D tests - 3d input, 1d indices -> 3d output
    IntPowerParameters sz0("sz0", 2, 10, 18, 2);      //2^10 to 2^18 in steps of 2
    PredefinedParameters sz1("sz1", {64, 4});
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

        ctx->setInputArray(0, in);
        ctx->setInputArray(1, indices);
        ctx->setInputArray(2, updates);
        ctx->setOutputArray(0, in);         //Needs to be inplace to avoid copy!
        ctx->markInplace(true);
        return ctx;
    };

    helper.runOperationSuit(&sa3d, generator3, batch3, "Scatter Update - 3d");
}


