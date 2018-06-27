//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"
#include <memory>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
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