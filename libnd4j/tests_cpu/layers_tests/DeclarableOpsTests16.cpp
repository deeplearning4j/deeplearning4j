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
                listI.at(e)->assign(rowOriginal);
                listE.at(e)->assign(rowReversed);
            }

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
     rgbs = np.random.uniform(0,1, 5*4*3 ).astype('float32').reshape([5,4,3])
     hsvs=np.apply_along_axis(lambda x: np.array(rgb_to_hsv(x[0],x[1],x[2])),2,rgbs)
     rgbs.ravel()
     hsvs.ravel()
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 5, 4, 3 }, {
         0.545678377f, 0.725874603f, 0.413571358f, 0.644941628f, 0.517642438f,
         0.890151322f, 0.461456001f, 0.0869259685f, 0.928968489f, 0.588904262f,
         0.54742825f, 0.684074104f, 0.52110225f, 0.761800349f, 0.486593395f,
         0.753103435f, 0.237176552f, 0.263826847f, 0.913557053f, 0.90049392f,
         0.290193319f, 0.46850124f, 0.965541422f, 0.148351923f, 0.674094439f,
         0.524110138f, 0.216262609f, 0.0361763388f, 0.2204483f, 0.279114306f,
         0.3721793f, 0.632020354f, 0.25007084f, 0.823592246f, 0.637001634f,
         0.30433768f, 0.0448598303f, 0.385092884f, 0.366362303f, 0.586083114f,
         0.218390301f, 0.931746006f, 0.978048146f, 0.762684941f, 0.00208298792f,
         0.91390729f, 0.505838513f, 0.875348926f, 0.428009957f, 0.367065936f,
         0.911922634f, 0.270003974f, 0.164243385f, 0.0581932105f, 0.313204288f,
         0.644775152f, 0.437950462f, 0.775881767f, 0.575452209f, 0.946475744f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 4, 3 }, {
         0.262831867f, 0.430244058f, 0.725874603f, 0.723622441f, 0.418478161f,
         0.890151322f, 0.740797927f, 0.906427443f, 0.928968489f, 0.717254877f,
         0.199753001f, 0.684074104f, 0.312434604f, 0.361258626f, 0.761800349f,
         0.991390795f, 0.685067773f, 0.753103435f, 0.163174023f, 0.682347894f,
         0.913557053f, 0.268038541f, 0.84635365f, 0.965541422f, 0.112067183f,
         0.679180562f, 0.674094439f, 0.540247589f, 0.870388806f, 0.279114306f,
         0.280050347f, 0.604331017f, 0.632020354f, 0.106776128f, 0.630475283f,
         0.823592246f, 0.490824632f, 0.883509099f, 0.385092884f, 0.75257351f,
         0.765611768f, 0.931746006f, 0.129888852f, 0.997870266f, 0.978048146f,
         0.849081645f, 0.446510047f, 0.91390729f, 0.685308874f, 0.597481251f,
         0.911922634f, 0.0834472676f, 0.784472764f, 0.270003974f, 0.396037966f,
         0.514242649f, 0.644775152f, 0.756701186f, 0.392005324f, 0.946475744f
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
    auto rgbs = NDArrayFactory::create<float>('c', { 5, 3, 4 }, {
         0.545678377f, 0.644941628f, 0.461456001f, 0.588904262f, 0.725874603f,
         0.517642438f, 0.0869259685f, 0.54742825f, 0.413571358f, 0.890151322f,
         0.928968489f, 0.684074104f, 0.52110225f, 0.753103435f, 0.913557053f,
         0.46850124f, 0.761800349f, 0.237176552f, 0.90049392f, 0.965541422f,
         0.486593395f, 0.263826847f, 0.290193319f, 0.148351923f, 0.674094439f,
         0.0361763388f, 0.3721793f, 0.823592246f, 0.524110138f, 0.2204483f,
         0.632020354f, 0.637001634f, 0.216262609f, 0.279114306f, 0.25007084f,
         0.30433768f, 0.0448598303f, 0.586083114f, 0.978048146f, 0.91390729f,
         0.385092884f, 0.218390301f, 0.762684941f, 0.505838513f, 0.366362303f,
         0.931746006f, 0.00208298792f, 0.875348926f, 0.428009957f, 0.270003974f,
         0.313204288f, 0.775881767f, 0.367065936f, 0.164243385f, 0.644775152f,
         0.575452209f, 0.911922634f, 0.0581932105f, 0.437950462f, 0.946475744f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 3, 4 }, {
         0.262831867f, 0.723622441f, 0.740797927f, 0.717254877f, 0.430244058f,
         0.418478161f, 0.906427443f, 0.199753001f, 0.725874603f, 0.890151322f,
         0.928968489f, 0.684074104f, 0.312434604f, 0.991390795f, 0.163174023f,
         0.268038541f, 0.361258626f, 0.685067773f, 0.682347894f, 0.84635365f,
         0.761800349f, 0.753103435f, 0.913557053f, 0.965541422f, 0.112067183f,
         0.540247589f, 0.280050347f, 0.106776128f, 0.679180562f, 0.870388806f,
         0.604331017f, 0.630475283f, 0.674094439f, 0.279114306f, 0.632020354f,
         0.823592246f, 0.490824632f, 0.75257351f, 0.129888852f, 0.849081645f,
         0.883509099f, 0.765611768f, 0.997870266f, 0.446510047f, 0.385092884f,
         0.931746006f, 0.978048146f, 0.91390729f, 0.685308874f, 0.0834472676f,
         0.396037966f, 0.756701186f, 0.597481251f, 0.784472764f, 0.514242649f,
         0.392005324f, 0.911922634f, 0.270003974f, 0.644775152f, 0.946475744f
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

    auto rgbs = NDArrayFactory::create<float>('c', { 4, 3 }, {
         0.545678377f, 0.725874603f, 0.413571358f, 0.644941628f, 0.517642438f,
         0.890151322f, 0.461456001f, 0.0869259685f, 0.928968489f, 0.588904262f,
         0.54742825f, 0.684074104f
        });
    auto expected = NDArrayFactory::create<float>('c', { 4, 3 }, {
         0.262831867f, 0.430244058f, 0.725874603f, 0.723622441f, 0.418478161f,
         0.890151322f, 0.740797927f, 0.906427443f, 0.928968489f, 0.717254877f,
         0.199753001f, 0.684074104f
        });

    auto actual = NDArrayFactory::create<float>('c', { 4, 3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_4) {
    auto rgbs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.545678377f, 0.644941628f, 0.461456001f, 0.588904262f, 0.725874603f,
         0.517642438f, 0.0869259685f, 0.54742825f, 0.413571358f, 0.890151322f,
         0.928968489f, 0.684074104f
        });
    auto expected = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.262831867f, 0.723622441f, 0.740797927f, 0.717254877f, 0.430244058f,
         0.418478161f, 0.906427443f, 0.199753001f, 0.725874603f, 0.890151322f,
         0.928968489f, 0.684074104f
        });

    auto actual = NDArrayFactory::create<float>('c', { 3, 4 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_5) {
    auto rgbs = NDArrayFactory::create<float>('c', { 3 }, {
        0.545678377f, 0.725874603f, 0.413571358f
        });
    auto expected = NDArrayFactory::create<float>('c', { 3 }, {
           0.262831867f, 0.430244058f, 0.725874603f
        });

    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(DeclarableOpsTests16, test_rgb_to_hsv_6) {
    auto rgbs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.545678377f, 0.644941628f, 0.461456001f, 0.588904262f, 0.725874603f,
         0.517642438f, 0.0869259685f, 0.54742825f, 0.413571358f, 0.890151322f,
         0.928968489f, 0.684074104f
        });
    auto hsvs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.262831867f, 0.723622441f, 0.740797927f, 0.717254877f, 0.430244058f,
         0.418478161f, 0.906427443f, 0.199753001f, 0.725874603f, 0.890151322f,
         0.928968489f, 0.684074104f
        });

    //get subarray 
    //get subarray
    NDArray subArrRgbs = rgbs.subarray({ NDIndex::all(), NDIndex::point(0) });
    NDArray expected = hsvs.subarray({ NDIndex::all(), NDIndex::point(0) });
    subArrRgbs.reshapei({ 3 });
    expected.reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrRgbs->printShapeInfo("subArrRgbs");
#endif
    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &subArrRgbs);
    ctx.setOutputArray(0, &actual);
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_1) {

    auto hsvs = NDArrayFactory::create<float>('c', { 5, 4, 3 }, {
         0.705504596f, 0.793608069f, 0.65870738f, 0.848827183f, 0.920532584f,
         0.887555957f, 0.72317636f, 0.563831031f, 0.773604929f, 0.269532293f,
         0.332347751f, 0.111181192f, 0.239250854f, 0.499201417f, 0.862712979f,
         0.0853395388f, 0.0810681432f, 0.226065159f, 0.851340771f, 0.602043271f,
         0.690895379f, 0.971996486f, 0.273846686f, 0.464318275f, 0.194078103f,
         0.219649255f, 0.616706491f, 0.847525477f, 0.653597355f, 0.700065672f,
         0.0299375951f, 0.184475258f, 0.274936169f, 0.196718201f, 0.179381892f,
         0.934476376f, 0.895766437f, 0.52967906f, 0.675635338f, 0.966644645f,
         0.770889699f, 0.556649387f, 0.13426739f, 0.899450243f, 0.817096591f,
         0.150202557f, 0.763557851f, 0.709604502f, 0.741747797f, 0.657703638f,
         0.167678103f, 0.828556478f, 0.615502477f, 0.478080243f, 0.447288662f,
         0.864299297f, 0.129833668f, 0.66402483f, 0.795475543f, 0.561332941f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 4, 3 }, {
         0.257768334f, 0.135951888f, 0.65870738f, 0.887555957f, 0.0705317783f,
         0.811602857f, 0.485313689f, 0.337422464f, 0.773604929f, 0.0883753772f,
         0.111181192f, 0.074230373f, 0.675155059f, 0.862712979f, 0.432045438f,
         0.226065159f, 0.21712242f, 0.207738476f, 0.690895379f, 0.274946465f,
         0.645954334f, 0.464318275f, 0.337166255f, 0.358530475f, 0.594427716f,
         0.616706491f, 0.481247369f, 0.700065672f, 0.242504601f, 0.661103036f,
         0.274936169f, 0.233327664f, 0.224217249f, 0.904251479f, 0.934476376f,
         0.766848235f, 0.675635338f, 0.317765447f, 0.54157777f, 0.556649387f,
         0.127534108f, 0.213413864f, 0.817096591f, 0.674227886f, 0.0821588641f,
         0.709604502f, 0.656080596f, 0.167780413f, 0.107076412f, 0.0573956046f,
         0.167678103f, 0.46964643f, 0.183820669f, 0.478080243f, 0.01761852f,
         0.129833668f, 0.0943436049f, 0.114806315f, 0.121884218f, 0.561332941f
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
    auto hsvs = NDArrayFactory::create<float>('c', { 5, 3, 4 }, {
         0.705504596f, 0.848827183f, 0.72317636f, 0.269532293f, 0.793608069f,
         0.920532584f, 0.563831031f, 0.332347751f, 0.65870738f, 0.887555957f,
         0.773604929f, 0.111181192f, 0.239250854f, 0.0853395388f, 0.851340771f,
         0.971996486f, 0.499201417f, 0.0810681432f, 0.602043271f, 0.273846686f,
         0.862712979f, 0.226065159f, 0.690895379f, 0.464318275f, 0.194078103f,
         0.847525477f, 0.0299375951f, 0.196718201f, 0.219649255f, 0.653597355f,
         0.184475258f, 0.179381892f, 0.616706491f, 0.700065672f, 0.274936169f,
         0.934476376f, 0.895766437f, 0.966644645f, 0.13426739f, 0.150202557f,
         0.52967906f, 0.770889699f, 0.899450243f, 0.763557851f, 0.675635338f,
         0.556649387f, 0.817096591f, 0.709604502f, 0.741747797f, 0.828556478f,
         0.447288662f, 0.66402483f, 0.657703638f, 0.615502477f, 0.864299297f,
         0.795475543f, 0.167678103f, 0.478080243f, 0.129833668f, 0.561332941f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 3, 4 }, {
         0.257768334f, 0.887555957f, 0.485313689f, 0.0883753772f, 0.135951888f,
         0.0705317783f, 0.337422464f, 0.111181192f, 0.65870738f, 0.811602857f,
         0.773604929f, 0.074230373f, 0.675155059f, 0.226065159f, 0.690895379f,
         0.464318275f, 0.862712979f, 0.21712242f, 0.274946465f, 0.337166255f,
         0.432045438f, 0.207738476f, 0.645954334f, 0.358530475f, 0.594427716f,
         0.700065672f, 0.274936169f, 0.904251479f, 0.616706491f, 0.242504601f,
         0.233327664f, 0.934476376f, 0.481247369f, 0.661103036f, 0.224217249f,
         0.766848235f, 0.675635338f, 0.556649387f, 0.817096591f, 0.709604502f,
         0.317765447f, 0.127534108f, 0.674227886f, 0.656080596f, 0.54157777f,
         0.213413864f, 0.0821588641f, 0.167780413f, 0.107076412f, 0.46964643f,
         0.01761852f, 0.114806315f, 0.0573956046f, 0.183820669f, 0.129833668f,
         0.121884218f, 0.167678103f, 0.478080243f, 0.0943436049f, 0.561332941f
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
    auto hsvs = NDArrayFactory::create<float>('c', { 4, 3 }, {
         0.705504596f, 0.793608069f, 0.65870738f, 0.848827183f, 0.920532584f,
         0.887555957f, 0.72317636f, 0.563831031f, 0.773604929f, 0.269532293f,
         0.332347751f, 0.111181192f
        });
    auto expected = NDArrayFactory::create<float>('c', { 4, 3 }, {
         0.257768334f, 0.135951888f, 0.65870738f, 0.887555957f, 0.0705317783f,
         0.811602857f, 0.485313689f, 0.337422464f, 0.773604929f, 0.0883753772f,
         0.111181192f, 0.074230373f
        });
    auto actual = NDArrayFactory::create<float>('c', { 4,3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(DeclarableOpsTests16, test_hsv_to_rgb_4) {
    auto hsvs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.705504596f, 0.848827183f, 0.72317636f, 0.269532293f, 0.793608069f,
         0.920532584f, 0.563831031f, 0.332347751f, 0.65870738f, 0.887555957f,
         0.773604929f, 0.111181192f
        });
    auto expected = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.257768334f, 0.887555957f, 0.485313689f, 0.0883753772f, 0.135951888f,
         0.0705317783f, 0.337422464f, 0.111181192f, 0.65870738f, 0.811602857f,
         0.773604929f, 0.074230373f
        });
    auto actual = NDArrayFactory::create<float>('c', { 3, 4 });

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

    auto hsvs = NDArrayFactory::create<float>('c', { 3 }, {
        0.705504596f, 0.793608069f, 0.65870738f
        });
    auto expected = NDArrayFactory::create<float>('c', { 3 }, {
           0.257768334f, 0.135951888f, 0.65870738f
        });

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

    auto hsvs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.705504596f, 0.848827183f, 0.72317636f, 0.269532293f, 0.793608069f,
         0.920532584f, 0.563831031f, 0.332347751f, 0.65870738f, 0.887555957f,
         0.773604929f, 0.111181192f
        });
    auto rgbs = NDArrayFactory::create<float>('c', { 3, 4 }, {
         0.257768334f, 0.887555957f, 0.485313689f, 0.0883753772f, 0.135951888f,
         0.0705317783f, 0.337422464f, 0.111181192f, 0.65870738f, 0.811602857f,
         0.773604929f, 0.074230373f
        });

    auto actual = NDArrayFactory::create<float>('c', { 3 });
    //get subarray 
   NDArray subArrHsvs = hsvs.subarray({ NDIndex::all(), NDIndex::point(0) });
    subArrHsvs.reshapei({ 3 });
    NDArray expected = rgbs.subarray({ NDIndex::all(), NDIndex::point(0) });
    expected.reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrHsvs->printShapeInfo("subArrHsvs");
#endif 

    Context ctx(1);
    ctx.setInputArray(0, &subArrHsvs);
    ctx.setOutputArray(0, &actual);
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}
