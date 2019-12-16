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
 // @author raver119@gmail.com
 //

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <array>


using namespace nd4j;


class DeclarableOpsTests16 : public testing::Test {
public:

    DeclarableOpsTests16() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests16, scatter_upd_1) {
    auto x = NDArrayFactory::create<float>('c', { 3 }, { 1.f, 1.f, 1.f });
    auto y = NDArrayFactory::create<int>(0);
    auto w = NDArrayFactory::create<float>(3.0f);
    auto e = NDArrayFactory::create<float>('c', { 3 }, { 3.f, 1.f, 1.f });

    nd4j::ops::scatter_upd op;
    auto result = op.execute({ &x, &y, &w }, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, scatter_upd_2) {

    NDArray x('c', { 10, 3 }, nd4j::DataType::FLOAT32);
    NDArray indices('c', { 2 }, { 2,5 }, nd4j::DataType::INT32);
    NDArray updates('c', { 2, 3 }, { 100,101,102,  200,201,202 }, nd4j::DataType::FLOAT32);
    NDArray e('c', { 10, 3 }, { 1,2,3, 4,5,6, 100,101,102, 10,11,12, 13,14,15, 200,201,202, 19,20,21, 22,23,24, 25,26,27, 28,29,30 }, nd4j::DataType::FLOAT32);

    x.linspace(1);

    nd4j::ops::scatter_upd op;
    auto result = op.execute({ &x, &indices, &updates }, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, scatter_upd_3) {

    NDArray x('c', { 10, 3 }, nd4j::DataType::FLOAT32);
    NDArray indices('c', { 2 }, { 20,5 }, nd4j::DataType::INT32);
    NDArray updates('c', { 2, 3 }, { 100,101,102,  200,201,202 }, nd4j::DataType::FLOAT32);
    NDArray output('c', { 10, 3 }, nd4j::DataType::FLOAT32);

    nd4j::ops::scatter_upd op;
    ASSERT_ANY_THROW(op.execute({ &x, &indices, &updates }, { &output }, {}, {}, { true, true }));
}

TEST_F(DeclarableOpsTests16, test_size_dtype_1) {
    auto x = NDArrayFactory::create<float>('c', { 3 }, { 1, 1, 1 });
    auto z = NDArrayFactory::create<float>(0.0f);
    auto e = NDArrayFactory::create<float>(3.0f);

    nd4j::ops::size op;
    auto status = op.execute({ &x }, { &z }, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests16, test_empty_noop_1) {
    auto z = NDArrayFactory::empty<Nd4jLong>();

    nd4j::ops::noop op;
    auto status = op.execute({}, { &z }, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_empty_noop_2) {
    auto z = NDArrayFactory::empty<Nd4jLong>();

    Context ctx(1);
    ctx.setOutputArray(0, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

    nd4j::ops::noop op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_svd_1) {
    auto x = NDArrayFactory::create<float>('c', { 3, 3 }, { 0.7787856f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f,0.50563407f, 0.89252293f, 0.5461209f });
    auto z = NDArrayFactory::create<float>('c', { 3 });

    nd4j::ops::svd op;
    auto status = op.execute({ &x }, { &z }, {}, { 0, 0, 16 }, {});

    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_hamming_distance_1) {
    auto x = NDArrayFactory::create<Nd4jLong>({ 37, 37, 37 });
    auto y = NDArrayFactory::create<Nd4jLong>({ 8723, 8723, 8723 });
    auto e = NDArrayFactory::create<Nd4jLong>(18);

    nd4j::ops::bits_hamming_distance op;
    auto result = op.execute({ &x, &y }, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, test_knn_mindistance_1) {
    auto input = NDArrayFactory::create<float>('c', { 512 });
    auto low = NDArrayFactory::create<float>('c', { 512 });
    auto high = NDArrayFactory::create<float>('c', { 512 });

    auto output = NDArrayFactory::create<float>(0.0f);

    input.linspace(1.0);
    low.linspace(1.0);
    high.linspace(1.0);

    nd4j::ops::knn_mindistance op;
    auto result = op.execute({ &input, &low, &high }, { &output }, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests16, test_empty_cast_1) {
    auto x = NDArrayFactory::create<bool>('c', { 1, 0, 2 });
    auto e = NDArrayFactory::create<Nd4jLong>('c', { 1, 0, 2 });

    nd4j::ops::cast op;
    auto result = op.execute({ &x }, {}, { 10 });
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests16, test_range_1) {
    nd4j::ops::range op;
    auto z = NDArrayFactory::create<float>('c', { 200 });

    Context ctx(1);
    ctx.setTArguments({ -1.0, 1.0, 0.01 });
    ctx.setOutputArray(0, &z);

    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_range_2) {
    nd4j::ops::range op;
    auto z = NDArrayFactory::create<float>('c', { 200 });

    double tArgs[] = { -1.0, 1.0, 0.01 };

    auto shapes = ::calculateOutputShapes2(nullptr, op.getOpHash(), nullptr, nullptr, 0, tArgs, 3, nullptr, 0, nullptr, 0);
    shape::printShapeInfoLinear("Result", shapes->at(0));
    ASSERT_TRUE(shape::shapeEquals(z.shapeInfo(), shapes->at(0)));

    delete shapes;
}

TEST_F(DeclarableOpsTests16, test_reverse_1) {
    std::vector<Nd4jLong> rows = { 3, 5, 7, 8, 9, 10, 119, 211 };
    std::vector<Nd4jLong> columns = { 6, 5, 10, 100, 153, 171, 635 };

    for (auto r : rows) {
        for (auto c : columns) {
            //nd4j_printf("Trying [%i, %i]\n", r, c);
            auto array = NDArrayFactory::create<float>('c', { r, c });
            auto exp = NDArrayFactory::create<float>('c', { r, c });
            auto reversed = NDArrayFactory::create<float>('c', { r, c });

            auto rowOriginal = NDArrayFactory::create<float>('c', { c });
            auto rowReversed = NDArrayFactory::create<float>('c', { c });

            for (int e = 0; e < c; e++) {
                rowOriginal.p(e, (float)e);
                rowReversed.p(c - e - 1, (float)e);
            }


            auto listI = array.allTensorsAlongDimension({ 1 });
            auto listE = exp.allTensorsAlongDimension({ 1 });

            for (int e = 0; e < r; e++) {
                listI->at(e)->assign(rowOriginal);
                listE->at(e)->assign(rowReversed);
            }

            delete listI;
            delete listE;

            nd4j::ops::reverse op;
            Nd4jLong axis = 1;
            auto status = op.execute({ &array }, { &reversed }, {}, { axis }, {});
            ASSERT_EQ(Status::OK(), status);

            ASSERT_EQ(exp, reversed);
        }
    }
}

TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_1) {
    /*
     test case generated by python colorsys and scaled to suit our needs
     from colorsys import *
     from random import *
     import numpy as np
     rgbs = np.array([randint(0,255) for x in range(0,3*4*5)]).reshape([5,4,3])
     hsvs=np.apply_along_axis(lambda x: np.array(rgb_to_hsv(x[0]/255,x[1]/255,x[2]/255))*np.array([360,1,1]),2,rgbs)
     rgbs.ravel()
     hsvs.ravel()
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 5, 4, 3 },
        {
           213.f, 220.f, 164.f, 121.f, 180.f, 180.f,  18.f, 245.f,  75.f, 235.f,  76.f,  74.f, 168.f,
            50.f, 233.f, 191.f, 132.f, 100.f, 207.f,  37.f, 245.f,  77.f, 250.f, 182.f, 111.f,  52.f,
            59.f, 193.f, 147.f, 137.f, 168.f, 103.f, 121.f,  48.f, 191.f, 187.f,  53.f,  82.f, 239.f,
           156.f,  37.f, 118.f, 244.f,  90.f,   7.f, 221.f,  98.f, 243.f,  12.f, 209.f, 192.f,   2.f,
           115.f, 205.f,  79.f, 247.f,  32.f,  70.f, 152.f, 180.f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 4, 3 },
        {
           6.75000000e+01f, 2.54545455e-01f, 8.62745098e-01f, 1.80000000e+02f,
           3.27777778e-01f, 7.05882353e-01f, 1.35066079e+02f, 9.26530612e-01f,
           9.60784314e-01f, 7.45341615e-01f, 6.85106383e-01f, 9.21568627e-01f,
           2.78688525e+02f, 7.85407725e-01f, 9.13725490e-01f, 2.10989011e+01f,
           4.76439791e-01f, 7.49019608e-01f, 2.89038462e+02f, 8.48979592e-01f,
           9.60784314e-01f, 1.56416185e+02f, 6.92000000e-01f, 9.80392157e-01f,
           3.52881356e+02f, 5.31531532e-01f, 4.35294118e-01f, 1.07142857e+01f,
           2.90155440e-01f, 7.56862745e-01f, 3.43384615e+02f, 3.86904762e-01f,
           6.58823529e-01f, 1.78321678e+02f, 7.48691099e-01f, 7.49019608e-01f,
           2.30645161e+02f, 7.78242678e-01f, 9.37254902e-01f, 3.19159664e+02f,
           7.62820513e-01f, 6.11764706e-01f, 2.10126582e+01f, 9.71311475e-01f,
           9.56862745e-01f, 2.90896552e+02f, 5.96707819e-01f, 9.52941176e-01f,
           1.74822335e+02f, 9.42583732e-01f, 8.19607843e-01f, 2.06600985e+02f,
           9.90243902e-01f, 8.03921569e-01f, 1.06883721e+02f, 8.70445344e-01f,
           9.68627451e-01f, 1.95272727e+02f, 6.11111111e-01f, 7.05882353e-01f
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,4,3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_2) {
    /*
      swapped_rgbs=rgbs.swapaxes(1,2).ravel()
      swapped_hsvs=hsvs.swapaxes(1,2).ravel()
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 5,3,4 },
        {
            213.f, 121.f,  18.f, 235.f, 220.f, 180.f, 245.f,  76.f, 164.f, 180.f,  75.f,  74.f, 168.f,
            191.f, 207.f,  77.f,  50.f, 132.f,  37.f, 250.f, 233.f, 100.f, 245.f, 182.f, 111.f, 193.f,
            168.f,  48.f,  52.f, 147.f, 103.f, 191.f,  59.f, 137.f, 121.f, 187.f,  53.f, 156.f, 244.f,
            221.f,  82.f,  37.f,  90.f,  98.f, 239.f, 118.f,   7.f, 243.f,  12.f,   2.f,  79.f,  70.f,
            209.f, 115.f, 247.f, 152.f, 192.f, 205.f,  32.f, 180.f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5,3,4 },
        {
           6.75000000e+01f, 1.80000000e+02f, 1.35066079e+02f, 7.45341615e-01f,
           2.54545455e-01f, 3.27777778e-01f, 9.26530612e-01f, 6.85106383e-01f,
           8.62745098e-01f, 7.05882353e-01f, 9.60784314e-01f, 9.21568627e-01f,
           2.78688525e+02f, 2.10989011e+01f, 2.89038462e+02f, 1.56416185e+02f,
           7.85407725e-01f, 4.76439791e-01f, 8.48979592e-01f, 6.92000000e-01f,
           9.13725490e-01f, 7.49019608e-01f, 9.60784314e-01f, 9.80392157e-01f,
           3.52881356e+02f, 1.07142857e+01f, 3.43384615e+02f, 1.78321678e+02f,
           5.31531532e-01f, 2.90155440e-01f, 3.86904762e-01f, 7.48691099e-01f,
           4.35294118e-01f, 7.56862745e-01f, 6.58823529e-01f, 7.49019608e-01f,
           2.30645161e+02f, 3.19159664e+02f, 2.10126582e+01f, 2.90896552e+02f,
           7.78242678e-01f, 7.62820513e-01f, 9.71311475e-01f, 5.96707819e-01f,
           9.37254902e-01f, 6.11764706e-01f, 9.56862745e-01f, 9.52941176e-01f,
           1.74822335e+02f, 2.06600985e+02f, 1.06883721e+02f, 1.95272727e+02f,
           9.42583732e-01f, 9.90243902e-01f, 8.70445344e-01f, 6.11111111e-01f,
           8.19607843e-01f, 8.03921569e-01f, 9.68627451e-01f, 7.05882353e-01f
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,3,4 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 1 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_3) {
    /*
      2D
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 8,3 },
        { 130.f,  61.f, 239.f, 117.f,  16.f, 168.f, 181.f, 223.f,   0.f,  49.f, 195.f, 195.f, 131.f,
       153.f,  78.f,  86.f,  21.f, 209.f, 101.f,  14.f, 107.f, 191.f,  98.f, 210.f });
    auto expected = NDArrayFactory::create<float>('c', { 8,3 },
        { 263.25842697f,   0.74476987f,   0.9372549f, 279.86842105f,
         0.9047619f,   0.65882353f,  71.30044843f,   1.f,
         0.8745098f, 180.f,   0.74871795f,   0.76470588f,
        77.6f,   0.49019608f,   0.6f, 260.74468085f,
         0.89952153f,   0.81960784f, 296.12903226f,   0.86915888f,
         0.41960784f, 289.82142857f,   0.53333333f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 8,3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_4) {
    /*
      2D
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });
    auto expected = NDArrayFactory::create<float>('c', { 3, 8 },
        { 263.25842697f, 279.86842105f,  71.30044843f, 180.f,
        77.6f, 260.74468085f, 296.12903226f, 289.82142857f,
         0.74476987f,   0.9047619f,   1.f,   0.74871795f,
         0.49019608f,   0.89952153f,   0.86915888f,   0.53333333f,
         0.9372549f,   0.65882353f,   0.8745098f,   0.76470588f,
         0.6f,   0.81960784f,   0.41960784f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 3, 8 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_5) {
    /*

    */
    auto rgbs = NDArrayFactory::create<float>('c', { 3 },
        { 213.f, 220.f, 164.f });
    auto expected = NDArrayFactory::create<float>('c', { 3 },
        { 6.75000000e+01, 2.54545455e-01, 8.62745098e-01 });


    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_6) {
    /*

    */
    auto rgbs = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });

    auto expected = NDArrayFactory::create<float>('c', { 3 },
        { 263.25842697f, 0.74476987f,  0.9372549f });

    //get subarray 
    std::unique_ptr<NDArray> subArrRgbs(rgbs.subarray({ NDIndex::all(), NDIndex::point(0) }));
    subArrRgbs->reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrRgbs->printShapeInfo("subArrRgbs");
#endif
    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, subArrRgbs.get());
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    subArrRgbs->printBuffer("subArrRgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_1) {
    /*
     using the same numbers of rgb_to_hsv_1 test
    */
    auto expected = NDArrayFactory::create<float>('c', { 5,4,3 },
        { 213.f, 220.f, 164.f, 121.f, 180.f, 180.f,  18.f, 245.f,  75.f, 235.f,  76.f,  74.f, 168.f,
        50.f, 233.f, 191.f, 132.f, 100.f, 207.f,  37.f, 245.f,  77.f, 250.f, 182.f, 111.f,  52.f,
        59.f, 193.f, 147.f, 137.f, 168.f, 103.f, 121.f,  48.f, 191.f, 187.f,  53.f,  82.f, 239.f,
       156.f,  37.f, 118.f, 244.f,  90.f,   7.f, 221.f,  98.f, 243.f,  12.f, 209.f, 192.f,   2.f,
       115.f, 205.f,  79.f, 247.f,  32.f,  70.f, 152.f, 180.f }
    );
    auto hsvs = NDArrayFactory::create<float>('c', { 5,4,3 },
        {
      6.75000000e+01f, 2.54545455e-01f, 8.62745098e-01f, 1.80000000e+02f,
       3.27777778e-01f, 7.05882353e-01f, 1.35066079e+02f, 9.26530612e-01f,
       9.60784314e-01f, 7.45341615e-01f, 6.85106383e-01f, 9.21568627e-01f,
       2.78688525e+02f, 7.85407725e-01f, 9.13725490e-01f, 2.10989011e+01f,
       4.76439791e-01f, 7.49019608e-01f, 2.89038462e+02f, 8.48979592e-01f,
       9.60784314e-01f, 1.56416185e+02f, 6.92000000e-01f, 9.80392157e-01f,
       3.52881356e+02f, 5.31531532e-01f, 4.35294118e-01f, 1.07142857e+01f,
       2.90155440e-01f, 7.56862745e-01f, 3.43384615e+02f, 3.86904762e-01f,
       6.58823529e-01f, 1.78321678e+02f, 7.48691099e-01f, 7.49019608e-01f,
       2.30645161e+02f, 7.78242678e-01f, 9.37254902e-01f, 3.19159664e+02f,
       7.62820513e-01f, 6.11764706e-01f, 2.10126582e+01f, 9.71311475e-01f,
       9.56862745e-01f, 2.90896552e+02f, 5.96707819e-01f, 9.52941176e-01f,
       1.74822335e+02f, 9.42583732e-01f, 8.19607843e-01f, 2.06600985e+02f,
       9.90243902e-01f, 8.03921569e-01f, 1.06883721e+02f, 8.70445344e-01f,
       9.68627451e-01f, 1.95272727e+02f, 6.11111111e-01f, 7.05882353e-01f
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,4,3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_2) {
    /*
      using the same numbers of hsv_to_rgb_2
    */
    auto expected = NDArrayFactory::create<float>('c', { 5,3,4 },
        { 213.f, 121.f,  18.f, 235.f, 220.f, 180.f, 245.f,  76.f, 164.f, 180.f,  75.f,  74.f, 168.f,
       191.f, 207.f,  77.f,  50.f, 132.f,  37.f, 250.f, 233.f, 100.f, 245.f, 182.f, 111.f, 193.f,
       168.f,  48.f,  52.f, 147.f, 103.f, 191.f,  59.f, 137.f, 121.f, 187.f,  53.f, 156.f, 244.f,
       221.f,  82.f,  37.f,  90.f,  98.f, 239.f, 118.f,   7.f, 243.f,  12.f,   2.f,  79.f,  70.f,
       209.f, 115.f, 247.f, 152.f, 192.f, 205.f,  32.f, 180.f }
    );
    auto hsvs = NDArrayFactory::create<float>('c', { 5,3,4 },
        {
      6.75000000e+01f, 1.80000000e+02f, 1.35066079e+02f, 7.45341615e-01f,
       2.54545455e-01f, 3.27777778e-01f, 9.26530612e-01f, 6.85106383e-01f,
       8.62745098e-01f, 7.05882353e-01f, 9.60784314e-01f, 9.21568627e-01f,
       2.78688525e+02f, 2.10989011e+01f, 2.89038462e+02f, 1.56416185e+02f,
       7.85407725e-01f, 4.76439791e-01f, 8.48979592e-01f, 6.92000000e-01f,
       9.13725490e-01f, 7.49019608e-01f, 9.60784314e-01f, 9.80392157e-01f,
       3.52881356e+02f, 1.07142857e+01f, 3.43384615e+02f, 1.78321678e+02f,
       5.31531532e-01f, 2.90155440e-01f, 3.86904762e-01f, 7.48691099e-01f,
       4.35294118e-01f, 7.56862745e-01f, 6.58823529e-01f, 7.49019608e-01f,
       2.30645161e+02f, 3.19159664e+02f, 2.10126582e+01f, 2.90896552e+02f,
       7.78242678e-01f, 7.62820513e-01f, 9.71311475e-01f, 5.96707819e-01f,
       9.37254902e-01f, 6.11764706e-01f, 9.56862745e-01f, 9.52941176e-01f,
       1.74822335e+02f, 2.06600985e+02f, 1.06883721e+02f, 1.95272727e+02f,
       9.42583732e-01f, 9.90243902e-01f, 8.70445344e-01f, 6.11111111e-01f,
       8.19607843e-01f, 8.03921569e-01f, 9.68627451e-01f, 7.05882353e-01f
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,3,4 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 1 });
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_3) {
    /*
      2D
    */
    auto expected = NDArrayFactory::create<float>('c', { 8,3 },
        { 130.f,  61.f, 239.f, 117.f,  16.f, 168.f, 181.f, 223.f,   0.f,  49.f, 195.f, 195.f, 131.f,
       153.f,  78.f,  86.f,  21.f, 209.f, 101.f,  14.f, 107.f, 191.f,  98.f, 210.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 8,3 },
        { 263.25842697f,   0.74476987f,   0.9372549f, 279.86842105f,
         0.9047619f,   0.65882353f,  71.30044843f,   1.f,
         0.8745098f, 180.f,   0.74871795f,   0.76470588f,
        77.6f,   0.49019608f,   0.6f, 260.74468085f,
         0.89952153f,   0.81960784f, 296.12903226f,   0.86915888f,
         0.41960784f, 289.82142857f,   0.53333333f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 8,3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_4) {
    /*
      2D
    */
    auto expected = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 3,8 },
        { 263.25842697f, 279.86842105f,  71.30044843f, 180.f,
        77.6f, 260.74468085f, 296.12903226f, 289.82142857f,
         0.74476987f,   0.9047619f,   1.f,   0.74871795f,
         0.49019608f,   0.89952153f,   0.86915888f,   0.53333333f,
         0.9372549f,   0.65882353f,   0.8745098f,   0.76470588f,
         0.6f,   0.81960784f,   0.41960784f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 3,8 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_5) {
    /*

    */
    auto expected = NDArrayFactory::create<float>('c', { 3 },
        { 213.f, 220.f, 164.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 3 },
        { 6.75000000e+01, 2.54545455e-01, 8.62745098e-01 });


    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_6) {

    auto expected = NDArrayFactory::create<double>('c', { 3 },
        { 130.0, 61.0, 239.0 });
    auto hsvs = NDArrayFactory::create<double>('c', { 3,8 },
        { 263.25842697, 279.86842105,  71.30044843, 180,
        77.6, 260.74468085, 296.12903226, 289.82142857,
         0.74476987,   0.9047619,   1.,   0.74871795,
         0.49019608,   0.89952153,   0.86915888,   0.53333333,
         0.9372549,   0.65882353,   0.8745098,   0.76470588,
         0.6,   0.81960784,   0.41960784,   0.82352941
        });

    //get subarray 
    std::unique_ptr<NDArray> subArrHsvs(hsvs.subarray({ NDIndex::all(), NDIndex::point(0) }));
    subArrHsvs->reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrHsvs->printShapeInfo("subArrHsvs");
#endif
    auto actual = NDArrayFactory::create<double>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, subArrHsvs.get());
    ctx.setOutputArray(0, &actual);
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    subArrHsvs->printBuffer("subArrHsvs ");
    actual.printBuffer("rgb ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}