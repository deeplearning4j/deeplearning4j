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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>


using namespace nd4j;


class DeclarableOpsTests13 : public testing::Test {
public:

    DeclarableOpsTests13() {
        //printf("\n");
        //fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests13, test_pow_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {2.f, 2.f, 2.f, 2.f});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 3});
    auto e = NDArrayFactory::create<float>('c', {2, 2}, {8.f, 8.f, 8.f, 8.f});

    nd4j::ops::Pow op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_1) {
    auto start = NDArrayFactory::create<int>(0);
    auto limit = NDArrayFactory::create<int>(0);

    nd4j::ops::range op;
    auto result = op.execute({&start, &limit}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_2) {

    nd4j::ops::range op;
    auto result = op.execute({}, {1.0, 1.0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_3) {

    nd4j::ops::range op;
    auto result = op.execute({}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_argmax_edge_1) {
    auto ctx = new Context(1);
    auto arr = NDArrayFactory::create_<float>('c', {1024,1});

    ctx->setInputArray(0, arr, true);
    ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong >('c', {1}), true);
    ctx->setInputArray(1, NDArrayFactory::create_<Nd4jLong >(0), true);   //Axis 0


    nd4j::ops::argmax op;
    auto result = op.execute(ctx);
    ASSERT_EQ(Status::OK(), result);

    //nd4j_printf("Done\n","");
    delete ctx;
}

TEST_F(DeclarableOpsTests13, test_add_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 768});
    auto y = NDArrayFactory::create<float>('c', {768});
    auto e = NDArrayFactory::create<float>('c', {1, 768});;
    y. assign(1.0f);
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(DeclarableOpsTests13, test_listdiff_1) {
    auto x = NDArrayFactory::create<int>('c', {4}, {0, 1, 2, 3});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 1});

    auto od = NDArrayFactory::create<int>('c', {2});
    auto oi = NDArrayFactory::create<int>('c', {2});

    nd4j::ops::listdiff op;
    auto result = op.execute({&x, &y}, {&od, &oi}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests13, test_greater_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 1});
    auto y = NDArrayFactory::create<float>('c', {1, 4});

    nd4j::ops::greater op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_eval_reduction_shape_1) {
    Nd4jLong axis = 0L;
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {4, 2});
    auto y = NDArrayFactory::create<Nd4jLong>('c', {1}, {axis});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {2}, {1, 2});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.execute({&x, &y}, {}, {}, {true});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, test_or_1) {

    NDArray x('c', {4}, {false, true, false, true}, nd4j::DataType::BOOL);
    NDArray y('c', {4}, {false, false, true, true}, nd4j::DataType::BOOL);
    NDArray e('c', {4}, {false, true, true, true}, nd4j::DataType::BOOL);

    NDArray z('c', {4}, nd4j::DataType::BOOL);

    x.applyPairwiseTransform(pairwise::Or, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_and_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, false, false, true});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::And, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_xor_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, true, true, false});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::Xor, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_1) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {1,2,3, 4, 5, 6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {1,-2,3, -4, 5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1.2,2.2,3.2,4.2,5.2,6.2});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_2) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {1, -2, 3, -4, 5, -6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {1, -2, 3, -4, 5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1.2, 0.01, 3.2, 0.01, 5.2, 0.01});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    //ASSERT_EQ(e, z);
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_3) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {-1, 2, -3, 4, -5, 6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {0.01, 2.2, 0.01, 4.2, 0.01, 6.2});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_1) {
    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {2}, {2, 3});
    auto cols = NDArrayFactory::create<int>('c', {5}, {0, 2, 1, 4, 3});
    auto vals = NDArrayFactory::create<double>('c', {5}, {10., 20., 30., 40., 50.});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp1 = NDArrayFactory::create<double>('c', {5,4}, {-1.846154, -1.846154, -1.846154, -1.846154, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {1});


    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    ASSERT_TRUE(exp1.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_2) {
    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {3}, {1,2,3});
    auto cols = NDArrayFactory::create<int>('c', {5}, {1, 2, 0, 4, 3});
    auto vals = NDArrayFactory::create<double>('c', {5}, {10., 20., 30., 40., 50.});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('c', {5,4}, {-0.622568, -0.622568, -0.622568, -0.622568, 1.846154, 1.846154, 1.846154, 1.846154, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {2});


    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_3) {
    auto data = NDArrayFactory::create<double>('c', {11, 5}, {0.3, 0.2625, 0.2674, 0.8604, 0.4803, 0.1096, 0.795, 0.5918, 0.2738, 0.952, 0.969, 0.8586, 0.8088, 0.5338, 0.5961, 0.7187, 0.463, 0.0867, 0.7748, 0.4802, 0.2493, 0.3227, 0.3064, 0.698, 0.7977, 0.7674, 0.168, 0.3107, 0.0217, 0.138, 0.8619, 0.8413, 0.5285, 0.9703, 0.6774, 0.2624, 0.4374, 0.1569, 0.1107, 0.0601, 0.4094, 0.9564, 0.5994, 0.8279, 0.3859, 0.6202, 0.7604, 0.0788, 0.0865, 0.7445, 0.6548, 0.3385, 0.0582, 0.6249, 0.7432});
    auto rows = NDArrayFactory::create<int>({0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99});
    auto cols = NDArrayFactory::create<int>({4, 3, 10, 8, 6, 7, 1, 5, 9, 4, 9, 8, 10, 2, 0, 6, 7, 3, 6, 8, 3, 9, 10, 1, 4, 0, 5, 10, 0, 4, 6, 8, 9, 2, 5, 7, 0, 10, 3, 1, 8, 9, 6, 7, 2, 7, 9, 3, 10, 0, 4, 2, 8, 1, 2, 8, 3, 10, 0, 4, 9, 1, 5, 5, 9, 0, 3, 10, 4, 8, 1, 2, 6, 2, 0, 3, 4, 1, 10, 9, 7, 10, 1, 3, 7, 4, 5, 2, 8, 6, 3, 4, 0, 9, 6, 5, 8, 7, 1});
    auto vals = NDArrayFactory::create<double>({0.6199614579042966, 0.19644097697184246, 0.13824979367331638, 0.01949900138247239, 0.008923198738222747, 0.008392793826291798, 0.0033348224714784204, 0.0026246189757042166, 0.0025733360563748838, 0.5877136110798608, 0.28250257562439585, 0.08098135424273815, 0.014862718272075049, 0.01219187321450782, 0.01152346362368888, 0.004243137936786281, 0.0034626999030188577, 0.0025185661029283168, 0.6777005651521399, 0.18321248222489303, 0.04018202465629351, 0.02941935889988646, 0.02164146250842832, 0.019898422145651618, 0.011683461395713935, 0.008439076090480863, 0.007823146926512332, 0.6770900431883232, 0.16617511239723026, 0.06039349887686468, 0.04650913399744179, 0.016886531410284355, 0.014591049666869658, 0.006407638669806174, 0.006074413005122801, 0.0058725787880570205, 0.6278185083409108, 0.235127797795446, 0.07023700015217448, 0.030885483448633774, 0.01229522088606573, 0.009238279699136107, 0.008219511168822047, 0.004303744819835723, 0.0018744536889749907, 0.7122603898978483, 0.07862620103245824, 0.07061257369349086, 0.06721483653169834, 0.028957853952131768, 0.01778978123182596, 0.01481713955181034, 0.005492728917348627, 0.0042284951913875955, 0.5266844101016999, 0.3304104787383107, 0.10930017433210941, 0.018514917515240075, 0.006969360999637938, 0.0063776901975396, 0.0010590388116165708, 6.526830884629785E-4, 3.1246215383067865E-5, 0.7176179284835663, 0.08741734015883978, 0.05927699083866909, 0.04663169573956976, 0.03287576269194147, 0.02993912340339554, 0.013365238657916641, 0.010616858763291145, 0.002259061262810172, 0.6891905160321706, 0.1397658294110526, 0.05438284759722162, 0.05437184733708826, 0.028683289714498808, 0.020986120697576355, 0.007218358114741088, 0.0032834770669826364, 0.002117714028667893, 0.6823873496503976, 0.1345267083671607, 0.08712863515505885, 0.04286621088946242, 0.02544804597749639, 0.01689343932533317, 0.007219134659004873, 0.0019232929717404616, 0.0016071830043453991, 0.6425809622897437, 0.18474464886441516, 0.10897036475298316, 0.03466939253836615, 0.013288054277817787, 0.005149178177380355, 0.0037974063158903518, 0.0037851733015991287, 0.0030148194818042273});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('c', {11, 5}, {-0.080205, -0.085862, 0.024045, 0.133551, -0.199896, -0.170597, 0.187301, 0.205824, -0.165268, 0.131228, 0.155135, 0.021446, 0.217583, -0.262873, -0.021075, 0.114537, 0.088023, -0.039205, 0.087984, -0.179565, -0.132683, 0.003677, 0.072081, -0.068737, 0.204481, 0.287223, -0.193989, 0.104569, -0.123401, -0.036368, 0.086745, 0.002961, -0.091327, 0.234853, 0.120270, -0.304006, 0.128305, -0.084867, -0.017550, -0.130837, -0.288569, 0.124679, 0.054078, -0.034187, -0.192599, 0.033196, 0.228182, -0.044972, -0.314217, 0.020287, 0.054427, -0.078887, -0.078246, -0.104543, 0.169803});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    //data.assign(1.0); //linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {11});

    //nd4j_printf("rows %lld, cols %lld, vals %lld, res full %lld\n", rows.lengthOf(), cols.lengthOf(), vals.lengthOf(), exp1.lengthOf());
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    //exp.printBuffer("Expect");
    //result->at(0)->printShapeInfo("Shape output");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_1) {
//    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {2}, {0, 1});
    auto cols = NDArrayFactory::create<int>('c', {4}, {0, 1, 1, 0});
    auto vals = NDArrayFactory::create<double>('c', {4}, {20., 30., 40., 50.});
    auto exp = NDArrayFactory::create<double>('c', {1,1}, {20.});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {1});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized1");
    ASSERT_TRUE(exp.equalsTo(result->at(2)));

    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_2) {
    auto rows = NDArrayFactory::create<int>('c', {4}, {0, 2, 2, 3});
    auto cols = NDArrayFactory::create<int>('c', {8}, {0, 1, 1, 0, 0, 1, 1, 1});
    auto vals = NDArrayFactory::create<double>('c', {8}, {20., 30., 40., 50., 120., 130., 140., 150.});
    auto exp = NDArrayFactory::create<double>('c', {1,5}, {20., 15., 15., 20., 20.});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {3});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized2");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    ASSERT_TRUE(exp.equalsTo(result->at(2)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_3) {
    auto rows = NDArrayFactory::create<int>('c', {12}, {0, 2, 3, 5, 7, 8, 9, 11, 12, 14, 18, 21});
    auto cols = NDArrayFactory::create<int>('c', {24}, {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 0, 2, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5});
    auto vals = NDArrayFactory::create<double>('c', {24}, {20., 30., 40., 50., 120., 130., 140., 150.,220., 230., 240., 250., 2120., 2130., 2140., 2150., 320., 330., 340., 350., 3120., 3130., 3140., 3150.});
    auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {11});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized3");
    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    //ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_4) {
    auto rows = NDArrayFactory::create<int>({0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
    auto cols = NDArrayFactory::create<int>({4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
    auto vals = NDArrayFactory::create<double>( {0.6200,    0.1964,    0.1382,    0.0195,    0.0089,    0.0084,    0.0033,    0.0026,    0.0026,    0.5877,    0.2825,    0.0810,    0.0149,    0.0122,    0.0115,    0.0042,    0.0035,    0.0025,    0.6777,    0.1832,    0.0402,    0.0294,    0.0216,    0.0199,    0.0117,    0.0084,    0.0078,    0.6771,    0.1662,    0.0604,    0.0465,    0.0169,    0.0146,    0.0064,    0.0061,    0.0059,    0.6278,    0.2351,    0.0702,    0.0309,    0.0123,    0.0092,    0.0082,    0.0043,    0.0019,    0.7123,    0.0786,    0.0706,    0.0672,    0.0290,    0.0178,    0.0148,    0.0055,    0.0042,    0.5267,    0.3304,    0.1093,    0.0185,    0.0070,    0.0064,    0.0011,    0.0007, 3.1246e-5,    0.7176,    0.0874,    0.0593,    0.0466,    0.0329,    0.0299,    0.0134,    0.0106,    0.0023,    0.6892,    0.1398,    0.0544,    0.0544,    0.0287,    0.0210,    0.0072,    0.0033,    0.0021,    0.6824,    0.1345,    0.0871,    0.0429,    0.0254,    0.0169,    0.0072,    0.0019,    0.0016,    0.6426,    0.1847,    0.1090,    0.0347,    0.0133,    0.0051,    0.0038,    0.0038,    0.0030});
    //auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
//    data.linspace(1);
    auto exp4 = NDArrayFactory::create<double>('c', {1, 108}, {0.6239, 0.1813, 0.1236, 0.03695, 0.00795, 0.03385, 0.0074, 0.0158, 0.0013, 0.0042, 0.0074, 0.3093, 0.2085, 0.051, 0.00895, 0.01605, 0.00245, 0.00705, 0.00125, 0.0021, 0.01605, 0.6022, 0.1615, 0.0233,
                                                0.0183, 0.0108, 0.0068, 0.0042, 0.0113, 0.00115, 0.1813, 0.00125, 0.0233, 0.65985, 0.0653, 0.0779, 0.03565, 0.05085, 0.03835, 0.02625, 0.6239, 0.3093, 0.0068, 0.0653, 0.2099, 0.0205, 0.0173, 0.0073,
                                                0.0171, 0.0089, 0.0158, 0.0113, 0.03835, 0.71495, 0.04775, 0.03615, 0.0089, 0.00275, 0.0021, 1.5623E-5, 0.00795, 0.00245, 0.6022, 0.0779, 0.0073, 0.5098, 0.0159, 0.00135, 1.5623E-5, 0.03385, 0.00705,
                                                0.02625, 0.0171, 0.71495, 0.06515, 0.01835, 0.00775, 0.00115, 0.03695, 0.051, 0.1615, 0.03565, 0.0205, 0.00275, 0.5098, 0.00775, 0.0055, 0.0026, 0.0013, 0.2085, 0.0183, 0.05085, 0.0173, 0.04775,
                                                0.00135, 0.06515, 0.0026, 0.35855, 0.1236, 0.00895, 0.0108, 0.65985, 0.2099, 0.03615, 0.0159, 0.01835, 0.0055, 0.35855});
//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {11});
    ASSERT_EQ(result->status(), Status::OK());
    auto res = result->at(2);
  //  res->printBuffer("Symmetrized4");
  //  exp4.printBuffer("Expected sym");
  //  nd4j_printf("Total res is {1, %lld}\n", res->lengthOf());
  //  nd4j_printf("Expected is {1, %lld}\n", exp4.lengthOf());

    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    ASSERT_TRUE(exp4.equalsTo(res));
    delete result;
}

TEST_F(DeclarableOpsTests13, CellContains_test_1) {

    auto corners = NDArrayFactory::create<double>( {0.5384,    0.5640,    0.3449,    0.5257,    0.5505});
    auto width = NDArrayFactory::create<double>({0.4306,    0.3960,    0.4639,    0.5040,    0.4904});
    auto point = NDArrayFactory::create<double>({0.3000,    0.2625,    0.2674,    0.8604,    0.4803});
    //auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
    //    data.linspace(1);

    //    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
    //    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    //    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::cell_contains op;
    auto result = op.execute({&corners, &width, &point}, {}, {5});
    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(result->at(0)->e<bool>(0));
    //result->at(2)->printBuffer("Symmetrized3");
    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    //ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_1) {

    NDArray input('c', {2,2,3}, {0,100,56, 17,220,5,  150,97,230, 255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {100,0,44, 208,5,220, 177,230,97,  2,255,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    auto results = op.execute({&input}, {0.5}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_2) {

    NDArray input('c', {2,2,3}, {0,100,56, 17,220,5,  150,97,230,   255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {4,100,0,  146,220,5, 97,123.8,230, 255,2,164.8}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    auto results = op.execute({&input}, {0.9}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_3) {

    NDArray input('c', {2,2,3}, {0,100,56,    17,220,5,          150,97,230,     255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {0.,84.,100., 5.,220.,122.0001,  229.8,97.,230., 255.,142.8002,2.}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    auto results = op.execute({&input}, {-0.9}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_4) {

    NDArray input('c', {2,3,2}, {0,17,   100,220, 56,5,   150,255, 97,2,   230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,3,2}, {100,208, 0,5,   44,220,  177,2,   230,255, 97,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    auto results = op.execute({&input}, {0.5}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_5) {

    NDArray input('c', {3,2,2}, {0,17, 150,255,   100,220, 97,2,  56,5, 230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {3,2,2}, {100,208, 177,2,  0,5, 230,255,   44,220, 97,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    auto results = op.execute({&input}, {0.5}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_1) {

    NDArray input('c', {2,2,3}, {0,100,56,  17,220,5,         150,97,230,    255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {50,100,78, 118.5,220,112.5,  190,163.5,230, 255,128.5,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {0.5}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_2) {

    NDArray input('c', {2,2,3}, {0,100,56,    17,220,5,          150,97,230,        255,2,13}, nd4j::DataType::DOUBLE);
    NDArray exp  ('c', {2,2,3}, {0.,100.,56., 12.279087,220.,0., 91.654228,0.,230., 255.,0.,11.087015}, nd4j::DataType::DOUBLE);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {10}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printIndexedBuffer("Result2");
//    exp.printIndexedBuffer("Expect2");

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_3) {

    NDArray input('c', {2,2,3}, {0,100,56,       17,220,5,       150,97,230,     255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {100.,100.,100., 220.,220.,220., 230.,230.,230., 255., 255., 255.}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {-10}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_4) {

    NDArray input('c', {2,3,2}, {0,17,   100,220,  56,5,   150,255,  97,2,   230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,3,2}, {50,118.5, 100,220, 78,112.5,  190,255, 163.5,128.5, 230,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {0.5}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_5) {

    NDArray input('c', {3,2,2}, {0,17,     150,255,  100,220,  97,2,        56,5,     230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {3,2,2}, {50,118.5, 190,255,  100,220,  163.5,128.5, 78,112.5, 230,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {0.5}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}


TEST_F(DeclarableOpsTests13, shift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(32);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, rshift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(512);
    e.assign(32);

    nd4j::ops::rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_shift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(32);
    e.assign(512);

    nd4j::ops::cyclic_shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_rshift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(512);
    e.assign(32);

    nd4j::ops::cyclic_rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, shift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, rshift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(512);
    y.assign(4);
    e.assign(32);

    nd4j::ops::rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_shift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::cyclic_shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_rshift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(512);
    y.assign(4);
    e.assign(32);

    nd4j::ops::cyclic_rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}
TEST_F(DeclarableOpsTests13, shift_bits_3) {
    auto x = NDArrayFactory::create<int>('c', {5, 5});
    auto y = NDArrayFactory::create<int>('c', {1, 5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_1) {

    NDArray x('c', {1, 2, 2, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 2} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {0, 0, 0, 0, 0, 0} , nd4j::DataType::INT32);

    NDArray exp('c', {8, 1, 1, 1, 3}, nd4j::DataType::FLOAT32);

    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_2) {

    NDArray x('c', {2,  2,4,3,  1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {0,0,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {24, 1,3,2, 1}, { 0, 2, 0, 8, 0, 0, 0, 26, 0, 32, 0, 0, 0, 3, 0, 9, 0, 0, 0, 27, 0, 33, 0, 0, 1,
                                        0, 7, 0, 0, 0, 25, 0, 31, 0, 0, 0, 0, 5, 0, 11, 0, 0, 0, 29, 0, 35, 0, 0, 0, 6,
                                        0, 12, 0, 0, 0, 30, 0, 36, 0, 0, 4, 0, 10, 0, 0, 0, 28, 0, 34, 0, 0, 0, 0, 14,
                                        0, 20, 0, 0, 0, 38, 0, 44, 0, 0, 0, 15, 0, 21, 0, 0, 0, 39, 0, 45, 0, 0, 13, 0,
                                        19, 0, 0, 0, 37, 0, 43, 0, 0, 0, 0, 17, 0, 23, 0, 0, 0, 41, 0, 47, 0, 0, 0, 18,
                                        0, 24, 0, 0, 0, 42, 0, 48, 0, 0, 16, 0, 22, 0, 0, 0, 40, 0, 46, 0, 0, 0}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_3) {

    NDArray x('c', {2,  2,4,3,  1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {1,1,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {24, 2,3,2, 1}, { 0, 0, 0, 0, 0, 0, 0, 14, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15,
                                        0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 19, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 37, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 47, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 18, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0,
                                        22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 46, 0, 0, 0, 0, 2, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 32,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                        0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 11, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 29, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 36, 0, 0,
                                        0, 0, 0, 0, 0, 0, 4, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_1) {

    NDArray x('c', {8, 1, 1, 1, 3}, nd4j::DataType::FLOAT32);

    NDArray blockShape('c', {3}, {2, 2, 2} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {0, 0, 0, 0, 0, 0} , nd4j::DataType::INT32);

    NDArray exp('c', {1, 2, 2, 2, 3}, nd4j::DataType::FLOAT32);

    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_2) {

    NDArray x('c', {24, 1,3,2, 1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {0,0,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {2,  2,4,3,  1}, {25, 2, 14, 61, 38, 50, 27, 4, 16, 63, 40, 52, 97, 74, 86, 133, 110, 122, 99, 76, 88, 135, 112, 124,
                                      31, 8, 20, 67, 44, 56, 33, 10, 22, 69, 46, 58, 103, 80, 92, 139, 116, 128, 105, 82, 94, 141, 118, 130}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_3) {

    NDArray x('c', {24, 2,3,2, 1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {1,1,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {2,  2,4,3,  1}, {193, 146, 170, 265, 218, 242, 195, 148, 172, 267, 220, 244, 55, 8, 32, 127, 80, 104, 57, 10, 34, 129, 82,
                                    106, 205, 158, 182, 277, 230, 254, 207, 160, 184, 279, 232, 256, 67, 20, 44, 139, 92, 116, 69, 22, 46, 141, 94, 118}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, mergemax_1) {

    NDArray x1('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray x3('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray e('c', {5, 5}, nd4j::DataType::FLOAT32);
    x1.assign(3);
    x2.assign(1);
    x3.assign(2);
    e.assign(3);


    nd4j::ops::mergemax op;
    auto result = op.execute({&x1, &x2, &x3}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, mergemax_2) {

    NDArray x1('c', {1, 3}, {0., 1, 2}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {1, 1}, {1.}, nd4j::DataType::FLOAT32);
    NDArray out('c', {1, 3}, {-1., -1, -1}, nd4j::DataType::FLOAT32);

    nd4j::ops::mergemax op;
    auto status = op.execute({&x1, &x2}, {&out}, {}, {}, {});

    ASSERT_EQ(20, status);
}



