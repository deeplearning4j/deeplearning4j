//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"
#include <memory>
#include <NDArray.h>
#include <NativeOps.h>
#include <types/float8.h>
#include "type_conversions.h"
using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class ConverTests : public testing::Test {
public:
    static const Nd4jLong rows = 3;
    static const int columns = 3;
};


//////////////////////////////////////////////////////////////////////
TEST_F(ConverTests, LongToFloat_Test) {
    Nd4jLong * A = new Nd4jLong[rows * columns]{
            0,2,7,
            2,36,35,
            3,30,17,
    };

    float * B = new float[rows * columns]{
            0.0,0.0,0.0,
            0.0,0.0,0.0,
            0.0,0.0,0.0,
    };

    float * Bexp = new float[rows * columns]{
            0.0,2.0,7.0,
            2.0,36.0,35.0,
            3.0,30.0,17.0,
    };


    Nd4jPointer * extras = new Nd4jPointer();
    NativeOps().convertTypes(extras, ND4J_INT64, (Nd4jPointer) A, 9L, ND4J_FLOAT32, (Nd4jPointer) B);

    for ( int i = 0; i < rows * columns; ++i){
        ASSERT_EQ(B[i], Bexp[i]);
    }


    delete[] A;
    delete[] B;
    delete[] Bexp;
}

TEST_F(ConverTests, QuartToInt_Test) {
    // Test that quarter to int works. (assuming quarter to float works)
    quarter data = quarter();

    // Loop over all possible binary representations
    for (unsigned char x = 0x00; x != 0xff; ++x) {
        data.x = x;
        float8 val = float8(data);
        float floatVal = (float) val;
        int expInt = static_cast<int>(floatVal);
        int foundInt = static_cast<int>(val);

        if (std::isnan(floatVal) || std::isinf(floatVal)){
            ASSERT_EQ(foundInt, INT32_MIN);
        } else {
            ASSERT_EQ(foundInt, expInt);
        }
    }
}

TEST_F(ConverTests, QuartToInt8_Test) {
    // Test that quarter to int works. (assuming quarter to float works)
    float8 src;
    int8 dst;
    src = float8();
    dst = src;
    ASSERT_EQ(dst.data, static_cast<int8_t>(0));

    src = dst;
    ASSERT_EQ(static_cast<float>(src), 0.0f);

    src.assign(16.0f);
    dst = src;


}
