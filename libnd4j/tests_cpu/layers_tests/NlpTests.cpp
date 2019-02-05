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

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <helpers/RandomLauncher.h>


using namespace nd4j;


class NlpTests : public testing::Test {
public:

    NlpTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(NlpTests, basic_sg_hs_test_1) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.01001f);
    exp1.assign(0.020005f);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto indices = NDArrayFactory::create<int>('c', {1}, {1});
    auto codes = NDArrayFactory::create<int8_t>('c', {1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.001);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(1L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result = op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row1 = syn1.subarray({NDIndex::point(1), NDIndex::all()});

    //row0->printIndexedBuffer("row0");
    //row1->printIndexedBuffer("row1");

    ASSERT_EQ(exp0, *row0);
    ASSERT_EQ(exp1, *row1);

    delete row0;
    delete row1;

    delete result;
}

TEST_F(NlpTests, basic_sg_hs_test_2) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.01f);
    exp1.assign(0.020005f);
    exp2.assign(0.019995f);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto indices = NDArrayFactory::create<int>('c', {2}, {1, 2});
    auto codes = NDArrayFactory::create<int8_t>('c', {2}, {0, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.001);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(1L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result = op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row1 = syn1.subarray({NDIndex::point(1), NDIndex::all()});
    auto row2 = syn1.subarray({NDIndex::point(2), NDIndex::all()});

    row0->printIndexedBuffer("row0");
    row1->printIndexedBuffer("row1");
    row2->printIndexedBuffer("row2");

    ASSERT_EQ(exp0, *row0);
    ASSERT_EQ(exp1, *row1);
    ASSERT_EQ(exp2, *row2);

    delete row0;
    delete row1;
    delete row2;

    delete result;
}

TEST_F(NlpTests, basic_sg_hs_test_3) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.01f);
    exp1.assign(0.020005f);
    exp2.assign(0.019995f);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto indices0 = NDArrayFactory::create<int>('c', {3}, {1, 2, 3});
    auto indices1 = NDArrayFactory::create<int>('c', {3}, {3, 1, 2});
    auto codes00 = NDArrayFactory::create<int8_t>('c', {3}, {0, 1, 1});
    auto codes01 = NDArrayFactory::create<int8_t>('c', {3}, {1, 0, 1});
    auto syn00 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn01 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn10 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn11 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    RandomGenerator rng(119L, 198L);
    RandomLauncher::fillUniform(rng, &syn00, 0.0, 1.0);
    RandomLauncher::fillUniform(rng, &syn10, 0.0, 1.0);

    syn01.assign(syn00);
    syn11.assign(syn10);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.001);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(1L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result0 = op.execute({&target, &ngStarter, &indices0, &codes00, &syn00, &syn10, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false}, true);
    auto result1 = op.execute({&target, &ngStarter, &indices1, &codes01, &syn01, &syn11, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false}, true);
    ASSERT_EQ(Status::OK(), result0->status());

    auto row00 = syn00.subarray({NDIndex::point(0), NDIndex::all()});
    auto row01 = syn01.subarray({NDIndex::point(0), NDIndex::all()});
    auto row1 = syn10.subarray({NDIndex::point(1), NDIndex::all()});
    auto row2 = syn11.subarray({NDIndex::point(1), NDIndex::all()});

    row00->printIndexedBuffer("syn00");
    row01->printIndexedBuffer("syn01");

    row1->printIndexedBuffer("syn10");
    row2->printIndexedBuffer("syn11");


    ASSERT_EQ(*row2, *row1);
    ASSERT_EQ(*row00, *row01);

    delete row00;
    delete row01;
    delete row1;
    delete row2;

    delete result0;
    delete result1;
}

TEST_F(NlpTests, basic_sg_hs_ns_test_1) {
    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::create<int>(1);
    auto indices = NDArrayFactory::create<int>('c', {5}, {1, 2, 3, 4, 5});
    auto codes = NDArrayFactory::create<int8_t>('c', {5}, {1, 1, 0, 1, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 150});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 150});
    auto syn1Neg = NDArrayFactory::create<float>('c', {100, 150});
    auto expTable = NDArrayFactory::create<float>('c', {1000});
    auto negTable = NDArrayFactory::create<float>('c', {1000});
    negTable.linspace(1.0);

    auto alpha = NDArrayFactory::create<double>(1.25);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(119L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result = op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {3}, {false}, true);
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(NlpTests, basic_sg_ns_test_1) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.01);

    auto target = NDArrayFactory::create<int>(1);
    auto ngStarter = NDArrayFactory::create<int>(3);
    auto indices = NDArrayFactory::empty<int>();
    auto codes = NDArrayFactory::empty<int8_t>();
    auto syn0 = NDArrayFactory::create<float>('c', {10, 10});
    auto syn1 = NDArrayFactory::empty<float>();
    auto syn1Neg = NDArrayFactory::create<float>('c', {10, 10});
    auto expTable = NDArrayFactory::create<float>('c', {1000});
    auto negTable = NDArrayFactory::create<float>('c', {1000});

    auto syn1Neg2 = NDArrayFactory::create<float>('c', {10, 10});

    syn0.assign(0.01);
    syn1.assign(0.02);
    syn1Neg.assign(0.03);
    syn1Neg2.assign(0.03);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.001);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(2L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result = op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {1}, {false}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row0 = syn0.subarray({NDIndex::point(1), NDIndex::all()});
    row0->printIndexedBuffer("row0");

    ASSERT_EQ(exp0, *row0);
    ASSERT_FALSE(syn1Neg2.equalsTo(syn1Neg, 1e-6));

    delete row0;

    delete result;
}

TEST_F(NlpTests, basic_cb_hs_test_1) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.0095f);
    exp1.assign(0.019875f);
    exp2.assign(0.02f);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto context = NDArrayFactory::create<int>('c', {3}, {0, 1, 2});
    auto indices = NDArrayFactory::create<int>('c', {2}, {4, 5});
    auto codes = NDArrayFactory::create<int8_t>('c', {2}, {1, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.025);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(2L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::cbow op;
    auto result = op.execute({&target, &ngStarter, &context, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {true}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row_s0_0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row_s0_1 = syn0.subarray({NDIndex::point(1), NDIndex::all()});
    auto row_s0_2 = syn0.subarray({NDIndex::point(2), NDIndex::all()});

    auto row_s1_4 = syn1.subarray({NDIndex::point(4), NDIndex::all()});
    auto row_s1_5 = syn1.subarray({NDIndex::point(5), NDIndex::all()});
    auto row_s1_6 = syn1.subarray({NDIndex::point(6), NDIndex::all()});

    row_s0_0->printIndexedBuffer("s0_0");
    row_s0_1->printIndexedBuffer("s0_1");
    row_s0_2->printIndexedBuffer("s0_2");

    row_s1_4->printIndexedBuffer("s1_4");
    row_s1_5->printIndexedBuffer("s1_5");
    row_s1_6->printIndexedBuffer("s1_6");

    ASSERT_EQ(exp0, *row_s0_0);
    ASSERT_EQ(exp0, *row_s0_1);
    ASSERT_EQ(exp0, *row_s0_2);

    ASSERT_EQ(exp1, *row_s1_4);
    ASSERT_EQ(exp1, *row_s1_5);

    ASSERT_EQ(exp2, *row_s1_6);

    delete row_s0_0;
    delete row_s0_1;
    delete row_s0_2;

    delete row_s1_4;
    delete row_s1_5;
    delete row_s1_6;

    delete result;
}

TEST_F(NlpTests, basic_cb_ns_test_1) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.0096265625);
    exp1.assign(0.01);
    exp2.assign(0.030125f);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::create<int>(6);
    auto context = NDArrayFactory::create<int>('c', {3}, {0, 1, 2});
    auto indices = NDArrayFactory::empty<int>();
    auto codes = NDArrayFactory::empty<int8_t>();
    auto syn0 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::create<float>('c', {100, 10});
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::create<float>('c', {100000});

    syn0.assign(0.01);
    syn1.assign(0.02);
    syn1Neg.assign(0.03);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.025);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(2L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::cbow op;
    auto result = op.execute({&target, &ngStarter, &context, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {0, 2}, {true}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row_s0_0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row_s0_1 = syn0.subarray({NDIndex::point(1), NDIndex::all()});
    auto row_s0_2 = syn0.subarray({NDIndex::point(2), NDIndex::all()});

    auto row_s1_4 = syn1.subarray({NDIndex::point(4), NDIndex::all()});
    auto row_s1_5 = syn1.subarray({NDIndex::point(5), NDIndex::all()});
    auto row_s1_6 = syn1Neg.subarray({NDIndex::point(6), NDIndex::all()});

    row_s0_0->printIndexedBuffer("s0_0");
    row_s0_1->printIndexedBuffer("s0_1");
    row_s0_2->printIndexedBuffer("s0_2");

    row_s1_4->printIndexedBuffer("s1_4");
    row_s1_5->printIndexedBuffer("s1_5");
    row_s1_6->printIndexedBuffer("s1_6");

    ASSERT_EQ(exp0, *row_s0_0);
    ASSERT_EQ(exp0, *row_s0_1);
    ASSERT_EQ(exp0, *row_s0_2);

    ASSERT_EQ(exp2, *row_s1_6);


    delete row_s0_0;
    delete row_s0_1;
    delete row_s0_2;

    delete row_s1_4;
    delete row_s1_5;
    delete row_s1_6;

    delete result;
}

TEST_F(NlpTests, test_sg_hs_batch_1) {
    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.01f);
    exp1.assign(0.020005f);
    exp2.assign(0.019995f);

    auto target = NDArrayFactory::create<int>('c', {2}, {0, 5});
    auto ngStarter = NDArrayFactory::empty<int>();
    auto indices = NDArrayFactory::create<int>('c', {2, 2}, {1, 2, 3, 4});
    auto codes = NDArrayFactory::create<int8_t>('c', {2, 2}, {0, 1, 1, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    auto alpha = NDArrayFactory::create<double>('c', {2}, {0.001, 0.024});
    auto randomValue = NDArrayFactory::create<Nd4jLong>('c', {2}, {1L, 3L});
    auto inferenceVector = NDArrayFactory::empty<float>();

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    nd4j::ops::skipgram op;
    auto result = op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row1 = syn1.subarray({NDIndex::point(1), NDIndex::all()});
    auto row2 = syn1.subarray({NDIndex::point(2), NDIndex::all()});

    row0->printIndexedBuffer("row0");
    row1->printIndexedBuffer("row1");
    row2->printIndexedBuffer("row2");

    ASSERT_TRUE(exp0.equalsTo(row0, 1e-6));
    ASSERT_TRUE(exp1.equalsTo(row1, 1e-6));
    ASSERT_TRUE(exp2.equalsTo(row2, 1e-6));

    delete row0;
    delete row1;
    delete row2;


    delete result;
}

TEST_F(NlpTests, test_cbow_hs_batch_1) {
    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto context = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2,  100, 101, 102});
    auto indices = NDArrayFactory::create<int>('c', {2, 2}, {4, 5, 40, 50});
    auto codes = NDArrayFactory::create<int8_t>('c', {2, 2}, {1, 1, 1, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {244, 10});
    auto syn1 = NDArrayFactory::create<float>('c', {244, 10});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {10000});
    auto negTable = NDArrayFactory::empty<float>();

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>('c', {2}, {0.025, 0.025});
    auto randomValue = NDArrayFactory::create<Nd4jLong>('c', {2}, {2L, 2L});
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::cbow op;
    auto result = op.execute({&target, &ngStarter, &context, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {true}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto exp0 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<float>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 10});

    exp0.assign(0.0095f);
    exp1.assign(0.019875f);
    exp2.assign(0.02f);

    auto row_s0_0 = syn0.subarray({NDIndex::point(0), NDIndex::all()});
    auto row_s0_1 = syn0.subarray({NDIndex::point(1), NDIndex::all()});
    auto row_s0_2 = syn0.subarray({NDIndex::point(2), NDIndex::all()});

    auto row_s1_4 = syn1.subarray({NDIndex::point(4), NDIndex::all()});
    auto row_s1_5 = syn1.subarray({NDIndex::point(5), NDIndex::all()});
    auto row_s1_6 = syn1.subarray({NDIndex::point(6), NDIndex::all()});

    row_s0_0->printIndexedBuffer("s0_0");
    row_s0_1->printIndexedBuffer("s0_1");
    row_s0_2->printIndexedBuffer("s0_2");

    row_s1_4->printIndexedBuffer("s1_4");
    row_s1_5->printIndexedBuffer("s1_5");
    row_s1_6->printIndexedBuffer("s1_6");

    ASSERT_EQ(exp0, *row_s0_0);
    ASSERT_EQ(exp0, *row_s0_1);
    ASSERT_EQ(exp0, *row_s0_2);

    ASSERT_EQ(exp1, *row_s1_4);
    ASSERT_EQ(exp1, *row_s1_5);

    ASSERT_EQ(exp2, *row_s1_6);

    delete row_s0_0;
    delete row_s0_1;
    delete row_s0_2;

    delete row_s1_4;
    delete row_s1_5;
    delete row_s1_6;

    delete result;
}