//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"
#include "DenseLayerTests.cpp"
#include "NDArrayTests.cpp"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}