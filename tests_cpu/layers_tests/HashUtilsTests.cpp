//
// Created by raver119 on 02.09.17.
//

#include "testlayers.h"
#include <helpers/helper_hash.h>

class HashUtilsTests : public testing::Test {

};


TEST_F(HashUtilsTests, TestEquality1) {
    std::string str("Conv2D");

    Nd4jLong hash1 = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
    ASSERT_EQ(-1637140380760460323L, hash1);
}



TEST_F(HashUtilsTests, TestEquality2) {
    std::string str("switch");

    Nd4jLong hash1 = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
    ASSERT_EQ(-1988317239813741487L, hash1);
}