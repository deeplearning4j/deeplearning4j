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
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests5 : public testing::Test {
public:

    DeclarableOpsTests5() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    auto exp = NDArrayFactory::create<double>('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
    x.linspace(1);
    x.reshapei('c', {3, 4, 5});

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {0, 2, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_0) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {1, 0, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_3) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {1, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_4) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {2, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_5) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {5, 4, 3}, {1.0, 21.0, 41.0, 6.0, 26.0, 46.0, 11.0, 31.0, 51.0, 16.0, 36.0, 56.0, 2.0, 22.0, 42.0, 7.0, 27.0, 47.0, 12.0, 32.0, 52.0, 17.0, 37.0, 57.0, 3.0, 23.0, 43.0, 8.0, 28.0, 48.0, 13.0, 33.0, 53.0, 18.0, 38.0, 58.0, 4.0, 24.0, 44.0, 9.0, 29.0, 49.0, 14.0, 34.0, 54.0, 19.0, 39.0, 59.0, 5.0, 25.0, 45.0, 10.0, 30.0, 50.0, 15.0, 35.0, 55.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 1, 0} shape");
//    x.printBuffer("{2, 1, 0} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {2, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TTS_bp_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 3});
    auto eps = NDArrayFactory::create<double>('c', {2, 4, 3});


    nd4j::ops::tile_to_shape_bp op;
    auto result = op.execute({&x, &eps}, {}, {2, 4, 3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("RES shape");
    x.printShapeInfo("EXP shape");
    z->printIndexedBuffer("RES output");
    ASSERT_TRUE(x.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_Rdiv_bp_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y = NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});


    nd4j::ops::reversedivide op_ff;
    auto result_ff = op_ff.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result_ff->status());

    auto z_ff = result_ff->at(0);
    ASSERT_TRUE(eps.isSameShape(z_ff));

    nd4j::ops::reversedivide_bp op_bp;
    auto result_bp = op_bp.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(Status::OK(), result_bp->status());

    auto z_bp = result_bp->at(0);
    ASSERT_TRUE(x.isSameShape(z_bp));

    delete result_ff;
    delete result_bp;
}


TEST_F(DeclarableOpsTests5, Test_Boolean_diff_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 1}, {1.0f});
    auto y = NDArrayFactory::create<double>(2.0f);

    nd4j::ops::less op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto paddings = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 0, 0});

    auto exp = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto paddings = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 0, 0});

    auto exp = NDArrayFactory::create<double>('c', {4, 1, 1, 1}, {1, 2, 3, 4});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto paddings = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 2, 0});

    auto exp = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto exp = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1) {
    auto x = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto crops = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1_1) {
    auto x = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_2) {
    auto x = NDArrayFactory::create<double>('c', {4, 1, 1, 1}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto crops = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3) {
    auto x = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto blocks = NDArrayFactory::create<double>('c', {2}, {2, 2});
    auto crops = NDArrayFactory::create<double>('c', {2, 2}, {0, 0, 2, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3_1) {
    auto x = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test1) {
    
    auto expected = NDArrayFactory::create<double>('c', {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3});
    auto output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test2) {
    
    auto expected = NDArrayFactory::create<double>('c', {3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3, 4});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test3) {
    
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3, 4, 2});
    auto output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test4) {
    
    auto expected = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3, 4, 2, 2});
    auto output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test1) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<double>('c', {2,2,1}, {3,2,3,2});

    auto expected = NDArrayFactory::create<double>('c', {2,2,3,2}, {19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test2) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<double>('c', {2,2,2}, {3,2,1,2, 0,1,0,1});

    auto expected = NDArrayFactory::create<double>('c', {2,2,2}, {23, 24, 11, 12, 3,  4, 3,  4});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test3) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<double>('c', {3}, {3,2,1});
    auto expected = NDArrayFactory::create<double>(24.);

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test4) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<double>('c', {2,3}, {3,2,1,0,2,1});
    auto expected = NDArrayFactory::create<double>('c',{2}, {24., 6});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test5) {

    auto input = NDArrayFactory::create<double>('c', {4}, {1,2,3,4});
    auto indices = NDArrayFactory::create<double>('c', {5,1}, {3,2,0,1,1});
    auto expected = NDArrayFactory::create<double>('c',{5}, {4.,3,1,2,2});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test6) {

    auto input = NDArrayFactory::create<double>('c', {4}, {1,2,3,4});
    std::vector<Nd4jLong> shape = {1};
    auto indices = NDArrayFactory::create<double>('c', shape, {2});
    auto expected = NDArrayFactory::create<double>(3.);

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test1) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {4}, {4,4,4,4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {4,  3,  2,  1,  5, 9,  8,  7,  6, 10, 14, 13, 12, 11, 15, 19, 18, 17, 16, 20, 24, 23, 22, 21, 25, 29, 28, 27, 26, 30, 34, 33, 32, 31, 35, 39, 38, 37, 36, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 1});
    auto output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test2) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {4}, {0,1,2,3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 12, 11, 13, 14, 15, 18, 17, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34, 35, 38, 37, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 51, 53, 54, 55, 58, 57, 56, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 1});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test3) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {3}, {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {2,  1,  3,  4,  5, 7,  6,  8,  9, 10, 12, 11, 13, 14, 15, 17, 16, 18, 19, 20, 23, 22, 21, 24, 25, 28, 27, 26, 29, 30, 33, 32, 31, 34, 35, 38, 37, 36, 39, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 0});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test4) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {5}, {1, 2, 1, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1, 22,  3, 24, 45, 6, 27,  8, 29, 50, 11, 32, 13, 34, 55, 16, 37, 18, 39, 60, 21,  2, 23,  4, 25, 26,  7, 28,  9, 30, 31, 12, 33, 14, 35, 36, 17, 38, 19, 40, 41, 42, 43, 44,  5, 46, 47, 48, 49, 10, 51, 52, 53, 54, 15, 56, 57, 58, 59, 20});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 2});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test5) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {5}, {1, 2, 4, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  7, 18,  9, 15, 6,  2, 13,  4, 10, 11, 12,  8, 14,  5, 16, 17,  3, 19, 20, 21, 27, 38, 29, 35, 26, 22, 33, 24, 30, 31, 32, 28, 34, 25, 36, 37, 23, 39, 40, 41, 47, 58, 49, 55, 46, 42, 53, 44, 50, 51, 52, 48, 54, 45, 56, 57, 43, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 2});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test6) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 2});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  2,  3,  4,  5, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 36, 37, 38, 39, 40, 21, 22, 23, 24, 25, 6,  7,  8,  9, 10, 31, 32, 33, 34, 35, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 56, 57, 58, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test7) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    std::vector<double> data = {3};
    auto seqLengths = NDArrayFactory::create<double>('c', {1}, data);    
    auto exp = NDArrayFactory::create<double>('c', {1, 5}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 0});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test8) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    std::vector<double> data = {1,0,1,0,1};
    auto seqLengths = NDArrayFactory::create<double>('c', {5}, data);    
    auto exp = NDArrayFactory::create<double>('c', {1, 5}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test9) {
    
    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    std::vector<double> data = {1,0,1,0,1};
    auto seqLengths = NDArrayFactory::create<double>('c', {5}, data);    
    auto exp = NDArrayFactory::create<double>('c', {5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 0});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test10) {
    
    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    std::vector<double> data = {3};
    auto seqLengths = NDArrayFactory::create<double>('c', {1}, data);    
    auto exp = NDArrayFactory::create<double>('c', {5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test11) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<double> data = {1, 0, 1, 0, 1};
    auto seqLengths = NDArrayFactory::create<double>('c', {5}, data);    
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 2});
    auto output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test12) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<double> data = {3};
    auto seqLengths = NDArrayFactory::create<double>('c', {1}, data);    
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 0});
    auto output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test13) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<double> data = {1};
    auto seqLengths = NDArrayFactory::create<double>('c', {1}, data);    
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {3, 0});
    auto output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_0) {
    auto x = NDArrayFactory::create<double>('c', {2, 6}, {1.0, 1.0, 1.0, 1.0, 11.0, 3.0, 1.0, 1.0, 1.0, 14.0, 5.0, 6.0});
    auto expV = NDArrayFactory::create<double>('c', {2, 1}, {11.0, 14.0});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 1}, {4, 3});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 0}); // without sorting

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);
/*
    v->printShapeInfo("topK_0: shape v");
    expV.printShapeInfo("topK_0: shape expV");

    i->printShapeInfo("topK_0: shape I");
    expI.printShapeInfo("topK_0: shape expI");

    v->printIndexedBuffer("topK_0: v");
    expV.printIndexedBuffer("topK_0: expV");
    i->printIndexedBuffer("topK_0: i");
    expI.printIndexedBuffer("topK_0: expI");
*/

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));
    // repeat res again
    for (int cases = 0; cases < 100; ++cases) {
        op.execute({&x}, {v, i}, {}, {1, 0}); // without sorting
    }
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    auto expV = NDArrayFactory::create<double>('c', {2, 1}, {11.0f, 14.0f});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 1}, {1, 0});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 0}); // without sorting

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("topK_1: shape v");
//    expV.printShapeInfo("topK_1: shape expV");

//    i->printShapeInfo("topK_1: shape I");
//    expI.printShapeInfo("topK_1: shape expI");

//    v->printIndexedBuffer("topK_1: v");
//    expV.printIndexedBuffer("topK_1: expV");
//    i->printIndexedBuffer("topK_1: i");
//    expI.printIndexedBuffer("topK_1: expI");


    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));
    // repeat res again
    for (int cases = 0; cases < 100; ++cases) {
        op.execute({&x}, {v, i}, {}, {1, 0}); // without sorting
    }
    delete result;
}

///////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_2) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );
// <<<14.>,<9.>>, <<21.>,<9.>>, <<14.>,<16.>>>
    auto expV = NDArrayFactory::create<double>('c', {2, 3, 1}, {14.0f, 9.0f,
                                         21.0f,
                                         9.0f, 14.0f,
                                         16.0f
                        }
    );

    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 3, 1 }, {2, 1, 0, 1, 2, 0});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

//    v->printIndexedBuffer("v");
//    expV.printIndexedBuffer("expV");
//    i->printIndexedBuffer("i");
//    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TopK_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );

    auto expV = NDArrayFactory::create<double>('c', {2, 3, 2}, {14.0f, 11.0f, 9.0f,
                                         7.0f, 21.0f, 15.0f,
                                         9.0f, 7.0f, 14.0f,
                                         13.0f, 16.0f, 13.5f
                        }
    );

    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 3, 2 }, {2, 0, 1, 3, 0, 3, 1,  3, 2, 1, 0, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

//    v->printIndexedBuffer("v");
//    expV.printIndexedBuffer("expV");
//    i->printIndexedBuffer("i");
//    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_4) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    auto expV = NDArrayFactory::create<double>('c', {2, 2}, {11.0f, 3.0f, 14.0f, 6.0f});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 2, 0, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

//    v->printIndexedBuffer("v");
//    expV.printIndexedBuffer("expV");
//    i->printIndexedBuffer("i");
//    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_5) {
    auto x = NDArrayFactory::create<double>('f', {2, 3}, {1.1, 11.1, 3.1, 14.2, 5.2, 6.2});
    auto expV = NDArrayFactory::create<double>('f', {2, 2}, {11.1, 3.1, 14.2, 6.2});
    auto expI = NDArrayFactory::create<Nd4jLong>('f', {2, 2}, {1, 2, 2, 0});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});
    for (Nd4jLong r = 0; r < 2; r++) {
        for (Nd4jLong c = 0; c < 3; c++)
            nd4j_printf("%f, ", x.e<double>(r,c));
        nd4j_printf("\n", "");
    }

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    x.printShapeInfo("shape of the source X");
//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");
    i->printIndexedBuffer("i");
    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0, 11.0, 3.0, 14.0, 5.0, 6.0});
    auto y = NDArrayFactory::create<Nd4jLong>('c', {2}, {1, 1});
    auto expV = NDArrayFactory::create<bool>('c', {2}, {true, false});

    nd4j::ops::in_top_k op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

     v->printShapeInfo("InTopK: shape v");
     expV.printShapeInfo("InTopK: shape expV");

     v->printIndexedBuffer("v");
     expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_2) {
    auto x = NDArrayFactory::create<double>('c', {6, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    auto y = NDArrayFactory::create<Nd4jLong>('c', {6}, {0, 0, 0, 0, 0, 0});
    auto expV = NDArrayFactory::create<bool>('c', {6}, {true, false, true, false, false, true});

    nd4j::ops::in_top_k op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    // v->printShapeInfo("InTopK: shape v");
    // expV.printShapeInfo("InTopK: shape expV");

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_3) {
    auto x = NDArrayFactory::create<double>('f', {6, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    auto y = NDArrayFactory::create<Nd4jLong>('f', {6}, {0, 0, 0, 0, 0, 0});
    auto expV = NDArrayFactory::create<bool>('f', {6}, {1, 0, 0, 0, 0, 0 });

    nd4j::ops::in_top_k op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    // v->printShapeInfo("InTopK: shape v");
    // expV.printShapeInfo("InTopK: shape expV");

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

///////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, Test_Moments_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    auto y = NDArrayFactory::create<double>('c', {3}, {0, 1, 2});
    //auto expV('f', {6}, {1, 0, 0, 0, 0, 0 });

    float expMean = 9.395833f;
    float expDeviation = 22.4579f;
//Mean 9.395833
//Deviance 22.4579

    float inf = 1.e-5f;

    nd4j::ops::moments op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

//    v->printIndexedBuffer("Result is ");
//    d->printIndexedBuffer("Result is ");

    ASSERT_TRUE(v->isScalar());
    ASSERT_NEAR(expMean, v->e<double>(0), inf);
    ASSERT_NEAR(expDeviation, d->e<double>(0), inf);

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_2) {
    NDArray x('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    NDArray expV('c', {4}, {11.833333, 7.6666665, 10.416667, 7.6666665});
    NDArray expD('c', {4}, {28.472221, 12.888889, 23.951387, 11.555554});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isVector());
    ASSERT_TRUE(d->isVector());

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );
    
    auto expV = NDArrayFactory::create<double>('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f, 
                                       8.5f, 11.f, 8.75f, 6.f, 
                                      18.5f, 6.f, 13.75f, 11.f});
    auto expD = NDArrayFactory::create<double>('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f,
                                       6.25f, 4.f, 27.5625f,  1.f,
                                       6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isMatrix());
    ASSERT_TRUE(d->isMatrix());

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_4) {
//    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0,  14.0, 5.0,
//                                       6.0,  9.0,   3.5, 7.0,
//                                     21.0, 3.0, 14.0, 15.0,
//                                      6.0, 9.0,  3.5,  7.0,
//                                      11.0, 13.0, 14.0, 5.0,
//                                      16.0,  9.0, 13.5, 7.0}
//    );
//   the fortran ordered matrix the same as C-ordered above
//
    auto x = NDArrayFactory::create<double>('f', {2, 3, 4}, {11.0f,  6.0f,  6.0f, 11.0f,
                                      21.0f, 16.0f,  3.0f,  9.0f,
                                       9.0f, 13.0f,  3.0f,  9.0f,
                                      14.0f,  3.5f,  3.5f, 14.0f,
                                      14.0f,  13.5f,  5.0f,  7.0f,
                                       7.0f,  5.0f, 15.0f,  7.0f
                                     }
    );


    auto expV = NDArrayFactory::create<double>('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f, 
                                       8.5f, 11.f, 8.75f, 6.f, 
                                      18.5f, 6.f, 13.75f, 11.f});
    auto expD = NDArrayFactory::create<double>('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f,
                                       6.25f, 4.f, 27.5625f,  1.f,
                                       6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isMatrix());
    ASSERT_TRUE(d->isMatrix());

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    // d->printIndexedBuffer("d");
    // expD.printIndexedBuffer("expD");

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test1) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3}, {40, 120, 200});

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test2) {
    
    auto input = NDArrayFactory::create<double>('c', {4, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(40.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test3) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test4) {
    
    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test5) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 4, 5, 6});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {75,  225,  375,  525, 675,  825,  975, 1125, 1275, 1425, 1575, 1725});

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test1) {
    
    auto input = NDArrayFactory::create<double>('c', {2, 2, 2});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test2) {
    
    auto input = NDArrayFactory::create<double>('c', {1, 3, 2});
    input.linspace(1);    

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test3) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 2, 1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);        

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test4) {
    auto input = NDArrayFactory::create<double>('c', {4});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);            

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test5) {
        
    auto input = NDArrayFactory::create<double>('c', {4,1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);                

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test6) {
        
    auto input = NDArrayFactory::create<double>('c', {4,1,1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);                    

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test7) {
        
    auto input = NDArrayFactory::create<double>('c', {1,4});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {1,4}, {1, 2, 3, 4});    

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);                    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, EmbeddingLookup_1) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 13, 23, 
                                      14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});
    
    auto y = NDArrayFactory::create<double>({1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 2.f, 2.f, 2.f});
    auto exp = NDArrayFactory::create<double>('c', {9, 4, 2}, {14, 24, 15, 25, 16, 26, 17, 27, 14, 24, 15, 25,
                                        16, 26, 17, 27, 14, 24, 15, 25, 16, 26, 17, 27,
                                        10, 20, 11, 21, 12, 22, 13, 23, 10, 20, 11, 21,
                                        12, 22, 13, 23, 10, 20, 11, 21, 12, 22, 13, 23,
                                        18, 28, 19, 29, 20, 30, 21, 31, 18, 28, 19, 29,
                                        20, 30, 21, 31, 18, 28, 19, 29, 20, 30, 21, 31});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup op;
    auto result = op.execute({&x, &y}, {}, {0});
    auto output = result->at(0);    
    // x.printShapeInfo("Input");
    // output->printShapeInfo("Output");
    // exp.printShapeInfo("Expected");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(exp.isSameShape(output));
    //output->printIndexedBuffer("Output");
    //exp.printIndexedBuffer("Expect");
    
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests5, EmbeddingLookup_2) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 30, 40, 50, 60, 
                                      70, 80, 90, 10, 11, 12, 
                                      13, 14, 15, 16, 17, 18, 
                                      19, 20, 21, 22, 23, 24});
                    //1,   0,   1,   0,   1,   0
    auto y = NDArrayFactory::create<double>({1.f, 0.f, 1.f, 0.f, 1.f, 0.f});
    auto exp = NDArrayFactory::create<double>('c', {6, 4, 2}, {90, 10, 11, 12, 13, 14,
                                        15, 16, 10, 20, 30, 40,
                                        50, 60, 70, 80, 90, 10,
                                        11, 12, 13, 14, 15, 16,
                                        10, 20, 30, 40, 50, 60,
                                        70, 80, 90, 10, 11, 12,
                                        13, 14, 15, 16, 10, 20,
                                        30, 40, 50, 60, 70, 80});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup op;
    auto result = op.execute({&x, &y}, {}, {0});
    auto output = result->at(0);    
    // x.printShapeInfo("Input");
    // output->printShapeInfo("Output");
    // exp.printShapeInfo("Expected");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(exp.isSameShape(output));
    // output->printIndexedBuffer("Output");
    // exp.printIndexedBuffer("Expect");
    
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests5, DynamicPartition_1) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 
                                      13, 23, 14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});
    
    auto y = NDArrayFactory::create<double>('c', {3, 4, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 
                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f 
                    }
    );
/*    auto y = NDArrayFactory::create<double>('c', {3, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 
                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f 
                    }
    );
*/
    int numPartition = 3;
    std::vector<NDArray> exp( { NDArrayFactory::create<double>('c', {6}, {10, 20, 11, 21, 12, 22}),
                                NDArrayFactory::create<double>('c', {8}, {18, 28, 19, 29, 20, 30, 21, 31}),
                                NDArrayFactory::create<double>('c', {10}, {13, 23, 14, 24, 15, 25, 16, 26, 17, 27})});

    nd4j::ops::dynamic_partition op;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // output->printIndexedBuffer("Output data> ");
        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicPartition_2) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    auto y = NDArrayFactory::create<double>('c', {2, 4}, {1, 2, 1, 2, 1, 2, 3, 0});

    std::vector<NDArray> exp( {NDArrayFactory::create<double>({-2.2f}),
                               NDArrayFactory::create<double>('c', {3}, {0.1f, 5.2f, -1.f}),
                               NDArrayFactory::create<double>('c', {3}, {-1.f, 4.3f, 7.4f}),
                               NDArrayFactory::create<double>({0.0f})});

    nd4j::ops::dynamic_partition op;
    int numPartition = 4;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // exp[e].printShapeInfo("Expected shape> ");
        // output->printIndexedBuffer("Output data> ");

        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}


TEST_F(DeclarableOpsTests5, DynamicPartition_3) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    auto y = NDArrayFactory::create<double>('c', {2, 4}, {0, 1, 0, 2, 0, 2, 3, 0});

    std::vector<NDArray> exp( {NDArrayFactory::create<double>({0.1f, 5.2f, -1.f, -2.2f}),
                               NDArrayFactory::create<double>({-1.f}),
                               NDArrayFactory::create<double>({4.3f, 7.4f}),
                               NDArrayFactory::create<double>({0.0f})});

    nd4j::ops::dynamic_partition op;
    int numPartition = 4;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        if (output)
        {
            // output->printShapeInfo("Output shape> ");
            // exp[e].printShapeInfo("Expected shape> ");
            // output->printIndexedBuffer("Output data> ");
        
            ASSERT_TRUE(exp[e].isSameShape(output));
            ASSERT_TRUE(exp[e].equalsTo(output));
        }
        else
        {
            ASSERT_TRUE(exp[e].lengthOf() == 0);
        }
    }

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_1) {
    
    auto x1 = NDArrayFactory::create<double>({1.f, 3.f, 5.f, 0.f});
    auto x2 = NDArrayFactory::create<double>({2.f, 4.f});
    auto y2 = NDArrayFactory::create<double>({-1.f, -1.f});
    auto y1 = NDArrayFactory::create<double>({0.1f, 5.2f, 4.3f, 7.4f});

    
    auto exp = NDArrayFactory::create<double>({7.4f, 0.1f, -1.f, 5.2f, -1.f, 4.3f});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_2) {
    
    auto x1 = NDArrayFactory::create<double>({1.f, 3.f});
    auto x2 = NDArrayFactory::create<double>({5.f, 0.f, 2.f, 4.f});
    auto y1 = NDArrayFactory::create<double>({-1.f, -1.f});
    auto y2 = NDArrayFactory::create<double>({0.1f, 5.2f, 4.3f, 7.4f});

    
    auto exp = NDArrayFactory::create<double>({5.2f, -1.f, 4.3f, -1.f, 7.4f, 0.1f});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test1) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);
    auto scale = NDArrayFactory::create<double>('c', {4});
    
    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {0,1});
    auto y = results->at(0);    
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test2) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);

    auto scale = NDArrayFactory::create<double>('c', {4});
    
    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.20347691,  1.20347691,  1.20347691,  1.20347691, 1.34829926,  1.34829926,  1.34829926,  1.34829926, 1.49312162,  1.49312162,  1.49312162,  1.49312162, 1.6379441 ,  1.6379441 ,  1.6379441 ,  1.6379441 , 1.78276646,  1.78276646,  1.78276646,  1.78276646, 1.92758882,  1.92758882,  1.92758882,  1.92758882, 2.0724113 ,  2.0724113 ,  2.0724113 ,  2.0724113 , 2.21723366,  2.21723366,  2.21723366,  2.21723366, 2.36205602,  2.36205602,  2.36205602,  2.36205602, 2.50687838,  2.50687838,  2.50687838,  2.50687838, 2.65170074,  2.65170074,  2.65170074,  2.65170074, 2.79652309,  2.79652309,  2.79652309,  2.79652309});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    auto y = results->at(0);    
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test3) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 4, 2, 3});
    x.linspace(1);
    
    auto scale = NDArrayFactory::create<double>('c', {4});
    
    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 4, 2, 3}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {1,1});
    auto y = results->at(0);    
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test4) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});    
    x.linspace(1);
    std::vector<Nd4jLong> shape = {4};
    auto scale = NDArrayFactory::create<double>('c', shape);    
    auto offset = NDArrayFactory::create<double>('c', shape);
    auto mean = NDArrayFactory::create<double>('c', shape);
    auto variance = NDArrayFactory::create<double>('c', shape);
    
    scale = 0.5;    
    offset = 2.;
    mean = 25.;
    variance = 5.;

    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {-3.36602688, -3.14244223, -2.91885757, -2.6952734 , -2.47168875, -2.24810457, -2.02451992, -1.80093551, -1.57735109, -1.35376668, -1.13018227, -0.90659785, -0.68301344, -0.45942879, -0.23584437, -0.01225996, 0.21132445,  0.43490887,  0.65849328,  0.88207781, 1.10566223,  1.32924664,  1.55283117,  1.77641559, 2.        ,  2.22358441,  2.44716883,  2.67075348, 2.89433765,  3.11792231,  3.34150672,  3.56509113, 3.78867555,  4.01225996,  4.23584461,  4.45942879, 4.68301344,  4.90659809,  5.13018227,  5.35376644, 5.57735109,  5.80093575,  6.02451992,  6.24810457, 6.47168875,  6.6952734 ,  6.91885757,  7.14244223});
    auto expBatchMean = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});
    auto expBatchVar = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {0,1});
    auto y = results->at(0);    
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test5) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});    
    x.linspace(1);
    std::vector<Nd4jLong> shape = {4};
    auto scale = NDArrayFactory::create<double>('c', shape);    
    auto offset = NDArrayFactory::create<double>('c', shape);
    auto mean = NDArrayFactory::create<double>('c', shape);
    auto variance = NDArrayFactory::create<double>('c', shape);
    
    scale = 0.5;    
    offset = 2.;
    mean = 25.;
    variance = 5.;

    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {-3.33992958e+00,  -3.11743259e+00,  -2.89493513e+00,  -2.67243814e+00, -2.44994116e+00,  -2.22744417e+00,  -2.00494719e+00,  -1.78244996e+00, -1.55995297e+00,  -1.33745599e+00,  -1.11495876e+00,  -8.92461777e-01, -6.69964790e-01,  -4.47467566e-01,  -2.24970579e-01,  -2.47359276e-03, 2.20023513e-01,   4.42520618e-01,   6.65017605e-01,   8.87514710e-01, 1.11001182e+00,   1.33250880e+00,   1.55500591e+00,   1.77750289e+00, 2.00000000e+00,   2.22249699e+00,   2.44499421e+00,   2.66749120e+00, 2.88998818e+00,   3.11248541e+00,   3.33498240e+00,   3.55747938e+00, 3.77997637e+00,   4.00247383e+00,   4.22497082e+00,   4.44746780e+00, 4.66996479e+00,   4.89246178e+00,   5.11495876e+00,   5.33745575e+00, 5.55995274e+00,   5.78244972e+00,   6.00494719e+00,   6.22744417e+00, 6.44994116e+00,   6.67243814e+00,   6.89493513e+00,   7.11743259e+00});
    auto expBatchMean = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});
    auto expBatchVar = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    auto y = results->at(0);    
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test1) {

    auto labels = NDArrayFactory::create<double>('c', {1, 3}, {1, 2, 4});
    auto predictions = NDArrayFactory::create<double>('c', {1, 3}, {2, 2, 4});
    auto expected = NDArrayFactory::create<double>('c', {5, 5}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions}, {}, {});
    auto output = results->at(0);
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test2) {

    auto labels = NDArrayFactory::create<double>('c', {1, 2}, {1, 2});
    auto predictions = NDArrayFactory::create<double>('c', {1, 2}, {0, 2});
    auto expected = NDArrayFactory::create<double>('c', {3, 3}, {0, 0, 0, 1, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions}, {}, {3});
    auto output = results->at(0);
    // output->printIndexedBuffer();


    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test3) {

    auto labels = NDArrayFactory::create<double>('c', {1, 2}, {1, 2});
    auto predictions = NDArrayFactory::create<double>('c', {1, 2}, {0, 2});
    auto weights = NDArrayFactory::create<double>('c', {1, 2}, {100, 200});
    auto expected = NDArrayFactory::create<double>('c', {3, 3}, {0, 0, 0, 100, 0, 0, 0, 0, 200});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions, &weights}, {}, {3});
    auto output = results->at(0);
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_1) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {0, 20, 30, 0, 50, 0, 
                                      70, 0, 90, 0, 11, 12, 
                                      13, 14, 15, 16, 17, 18, 
                                      19, 0, 21, 22, 23, 24});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<double>(0), 0.25);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_2) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {5.5, 0., 0.3, 5.5, 8.6, 0., 0., 0.4});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<double>(0), 0.375);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_3) {
    
    auto x = NDArrayFactory::create<double>('f', {2, 2, 2}, {5.5, 0., 0.3, 5.5, 8.6, 0., 0., 0.4});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<float>(0), 0.375);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, XWPlusB_1) {

    auto x = NDArrayFactory::create<double>('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});
    auto y = NDArrayFactory::create<double>('c', {3,2}, { 11.f,  3.f, 4.f,  5.f, 6.f,  2.f});
    auto b = NDArrayFactory::create<double>({100.f, 200.f});

    auto exp = NDArrayFactory::create<double>('c', {2,2}, {173.f, 264.f, 310.f, 279.f});

    nd4j::ops::xw_plus_b op;
    auto result = op.execute({&x, &y, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, StopGradient_1) {

    auto x = NDArrayFactory::create<double>('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // x.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // x.printIndexedBuffer("Expected res>");    

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, StopGradient_2) {

    auto x = NDArrayFactory::create<double>('f', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // x.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // x.printIndexedBuffer("Expected res>");    

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test1) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test2) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-3.05095e+00,-3.04946e+00,-5.00705e+00, -5.09458e-02,-7.04946e+00,-7.04851e-03, -6.05095e+00,-4.94556e-02,-8.00705e+00, -3.04859e+00,-1.30000e+01,-3.04859e+00, -1.50486e+01,-2.37286e-06,-1.70486e+01, -4.85876e-02,-1.60000e+01,-4.85874e-02, -2.10000e+01,-3.04859e+00,-2.51269e+01, -7.96007e-10,-2.50486e+01,-2.12693e+00, -2.40000e+01,-4.85874e-02,-1.26928e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {1});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test3) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {2});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test5) {

    auto input = NDArrayFactory::create<double>('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3}, {-2.16985, -0.16985, -3.16985, -1.31507, -6.31507, -0.31507, -9.31335, -1.31335, -0.31335});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test6) {

    auto input = NDArrayFactory::create<double>('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3}, {-3.05095,-3.04946,-7.12773, -0.05095,-7.04946,-2.12773, -6.05095,-0.04946,-0.12773});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test7) {

    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test8) {

    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test9) {

    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test10) {

    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test11) {

    auto input = NDArrayFactory::create<double>('c', {5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {});
    auto z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2}, {1,2,3,4});
    auto epsilon = NDArrayFactory::create<double>('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});    
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {-0.07311,0.02689, -0.07311,0.02689});
    
    nd4j::ops::log_softmax_bp op;
    auto  results = op.execute({&input, &epsilon}, {}, {});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test2) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2}, {1,2,3,4});
    auto epsilon = NDArrayFactory::create<double>('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});    
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {-0.17616, -0.17616, 0.02384,  0.02384});
    
    nd4j::ops::log_softmax_bp op;
    auto  results = op.execute({&input, &epsilon}, {}, {0});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ELU_1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto exp     = NDArrayFactory::create<double>('c', {2, 2, 2}, { -0.63212055,  2. , 1.5, -0.753403, 1.,   2.,  2.,   1.});
    auto res     = NDArrayFactory::create<double>('c', {2, 2, 2});
    
    input.applyTransform(transform::ELU, &res);

    ASSERT_TRUE(res.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, L2_Loss_1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    double exp(9.605);
    
    nd4j::ops::l2_loss op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(output->isScalar());
    output->printIndexedBuffer("L2_Loss output");
    ASSERT_EQ(output->e<double>(0), exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoisonLoss_1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto targets = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.3678794, 5.389056, 2.981689, 1.6465969, 1.7182817, 5.389056, 5.389056, 1.7182817});
    
    nd4j::ops::log_poison_loss op;
    auto results = op.execute({&targets, &input}, {}, {});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoisonLoss_2) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto targets = NDArrayFactory::create<double>('c', {2, 2, 2}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2}, {3.0196857, 4.0408626, 2.1334953, 3.6984034, 1.3700882, 4.0408626, 4.0408626, 1.3700882});
 
    nd4j::ops::log_poison_loss op;
    auto results = op.execute({&targets, &input}, {}, {1});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_1) {

    auto means   = NDArrayFactory::create<double>('c', {2, 3, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {2, 3, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(2.0);

    auto expMeans = NDArrayFactory::create<double>('c', {2, 3, 4}, {
                                                 5.5,   1.5,     7.,  2.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                10.5,   1.5,     7.,  7.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                 5.5,   6.5,     7.,  2.5,
                                                  8.,   4.5,   6.75,  3.5});

    auto expDeviance = NDArrayFactory::create<double>('c', {2, 3, 4}, {
                                                -19.75,     4.25,       -37.,   1.25,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -94.75,     4.25,       -37.,  -43.75,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -19.75,   -30.75,       -37.,   1.25,
                                                  -51.,   -10.75,   -33.8125,  -3.75});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);    
    auto outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_2) {

    auto means   = NDArrayFactory::create<double>('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(12.0);

    auto expMeans = NDArrayFactory::create<double>('c', {3, 2, 4}, { 0.9166667,     0.25,  1.1666667, 0.4166667,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                                    1.75,     0.25,  1.1666667,      1.25,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                               0.9166667, 1.0833334, 1.1666667, 0.4166667,
                                               1.3333334,      0.75,     1.125, 0.5833334});

    auto expDeviance = NDArrayFactory::create<double>('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);    
    auto outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_3) {

    auto means   = NDArrayFactory::create<double>('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(12.0);
    double shift = 10.0;
    auto expMeans = NDArrayFactory::create<double>('c', {3, 2, 4}, { 10.9166667,     10.25,  11.1666667, 10.4166667,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                                    11.75,     10.25,  11.1666667,      11.25,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                               10.9166667, 11.0833334, 11.1666667, 10.4166667,
                                               11.3333334,      10.75,     11.125, 10.5833334});

    auto expDeviance = NDArrayFactory::create<double>('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {shift}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);    
    auto outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

