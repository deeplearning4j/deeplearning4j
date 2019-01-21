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
// Created by raver119 on 11.10.2017.
//

#include "testlayers.h"
#include <vector>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpTuple.h>
#include <ops/declarable/OpRegistrator.h>
#include <GraphExecutioner.h>
#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include <MmulHelper.h>

using namespace nd4j;
using namespace nd4j::ops;

class OneOffTests : public testing::Test {
public:

};

TEST_F(OneOffTests, test_avg_pool_3d_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/avg_pooling3d.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_non2d_0A_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_0A.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_assert_scalar_float32_1) {
    nd4j::ops::Assert op;
    nd4j::ops::identity op1;
    nd4j::ops::noop op2;
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scalar_float32.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_pad_1D_1) {
    auto e = NDArrayFactory::create<float>('c', {7}, {10.f,0.778786f, 0.801198f, 0.724375f, 0.230894f, 0.727141f,10.f});
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/pad_1D.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(4));

    auto z = graph->getVariableSpace()->getVariable(4)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);
    delete graph;
}
/*
TEST_F(OneOffTests, test_scatter_nd_update_1) {

    auto e = NDArrayFactory::create<float>('c', {10, 7}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.20446908f, 0.37918627f, 0.99792874f, 0.71881700f, 0.18677747f,
                                                    0.78299069f, 0.55216062f, 0.40746713f, 0.92128086f, 0.57195139f, 0.44686234f, 0.30861020f, 0.31026053f, 0.09293187f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.95073712f, 0.45613325f, 0.95149803f, 0.88341522f, 0.54366302f, 0.50060666f, 0.39031255f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scatter_nd_update.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);

    delete graph;
}
 */

TEST_F(OneOffTests, test_conv2d_nhwc_failed_1) {
    auto e = NDArrayFactory::create<float>('c', {1, 5, 5, 6}, {0.55744928f, 0.76827729f, 1.09401524f, 0.00000000f, 0.00000000f, 0.00000000f, 0.56373537f, 0.90029907f, 0.78997850f, 0.00000000f, 0.00000000f, 0.00000000f, 0.14252824f, 0.95961076f, 0.87750554f, 0.00000000f, 0.00000000f, 0.00000000f, 0.44874173f, 0.99537718f, 1.17154264f, 0.00000000f, 0.00000000f, 0.00000000f, 0.60377145f, 0.79939061f, 0.56031001f, 0.00000000f, 0.00000000f, 0.00000000f, 0.52975273f, 0.90678585f, 0.73763013f, 0.00000000f, 0.00000000f, 0.00000000f, 0.22146404f, 0.82499605f, 0.47222072f, 0.00000000f, 0.00000000f, 0.00000000f, 0.42772964f, 0.39793295f, 0.71436501f, 0.00000000f, 0.00000000f, 0.00000000f, 0.48836520f, 1.01658893f, 0.74419701f, 0.00000000f, 0.00000000f, 0.00000000f, 0.78984612f, 0.94083673f, 0.83841157f, 0.00000000f, 0.00000000f, 0.00000000f, 0.40448499f, 0.67732805f, 0.75499672f, 0.00000000f, 0.00000000f, 0.00000000f, 0.43675962f, 0.79476535f, 0.72976631f, 0.00000000f, 0.00000000f, 0.00000000f, 0.58808053f, 0.65222591f, 0.72552216f, 0.00000000f, 0.00000000f, 0.00000000f, 0.37445742f, 1.22581339f, 1.05341125f, 0.00000000f, 0.00000000f, 0.00000000f, 0.30095795f, 0.59941679f, 0.63323414f, 0.00000000f, 0.00000000f, 0.00000000f, 0.24199286f, 1.02546394f, 0.69537812f, 0.00000000f, 0.00000000f, 0.00000000f, 0.23628944f, 0.90791851f, 1.01209974f, 0.00000000f, 0.00000000f, 0.00000000f, 0.62740159f, 0.56518674f, 0.76692569f, 0.00000000f, 0.00000000f, 0.00000000f, 0.13327584f, 0.32628393f, 0.10280430f, 0.00000000f, 0.00000000f, 0.00000000f, 0.42691272f, 0.25625113f, 0.30524066f, 0.00000000f, 0.00000000f, 0.00000000f, 0.17797673f, 0.84179950f, 0.80061519f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00199084f, 0.51838887f, 0.43932241f, 0.00000000f, 0.00000000f, 0.00000000f, 0.16684581f, 0.50822425f, 0.48668745f, 0.00000000f, 0.00000000f, 0.00000000f, 0.16749343f, 0.93093169f, 0.86871749f, 0.00000000f, 0.00000000f, 0.00000000f, 0.17486368f, 0.44460732f, 0.44499981f, 0.00000000f, 0.00000000f, 0.00000000f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/channels_last_b1_k2_s1_d1_SAME_crelu.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(9));

    auto z = graph->getVariableSpace()->getVariable(9)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_1) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_close_sz1_float32_nodynamic_noname_noshape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(5));

    auto z = graph->getVariableSpace()->getVariable(5)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_2) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_split_sz1_float32_nodynamic_noname_noshape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_3) {
    auto e = NDArrayFactory::create<int>('c', {3, 2, 3}, {7, 2, 9, 4, 3, 3, 8, 7, 0, 0, 6, 8, 7, 9, 0, 1, 1, 4});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_stack_sz3-1_int32_dynamic_name_shape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(15));

    auto z = graph->getVariableSpace()->getVariable(15)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_4) {
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 3}, {4, 3, 1, 1, 1, 0});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_unstack_sz1_int64_nodynamic_noname_shape2-3.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(11));

    auto z = graph->getVariableSpace()->getVariable(11)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_assert_4) {
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 1, 1, 1});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/assert_type_rank2_int64.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));

    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

// TEST_F(OneOffTests, test_cond_true_1) {
//     auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});

//     auto graph = GraphExecutioner::importFromFlatBuffers("./resources/cond_true.fb");
//     ASSERT_TRUE(graph != nullptr);

//     graph->printOut();


//     Nd4jStatus status = GraphExecutioner::execute(graph);
//     ASSERT_EQ(Status::OK(), status);
//     ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

//     auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
//     ASSERT_TRUE(z != nullptr);

//     z->printIndexedBuffer("z buffer");

//     ASSERT_EQ(e, *z);

//     delete graph;
// }

/*
TEST_F(OneOffTests, test_cond_false_1) {
    auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/cond_false.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z buffer");

    ASSERT_EQ(e, *z);

    delete graph;
}
*/

TEST_F(OneOffTests, test_identity_n_2) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    nd4j::ops::identity_n op;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/identity_n_2.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1, 1));

    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_non2d_1) {
    auto e = NDArrayFactory::create<float>('c', {1, 1}, {5.42746449f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_1.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(3));

    auto z = graph->getVariableSpace()->getVariable(3)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);


    delete graph;
}

TEST_F(OneOffTests, test_reduce_all_1) {
    auto e = NDArrayFactory::create<bool>('c', {1, 4}, {true, false, false, false});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_all_rank2_d0_keep.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(2));
    auto in = graph->getVariableSpace()->getVariable(2)->getNDArray();


    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);


    delete graph;
}