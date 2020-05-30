/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
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
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//


#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include "testlayers.h"
#include <graph/Stash.h>
#include <helpers/BitwiseUtils.h>
#include <bitset>

using namespace sd;

class StringTests : public testing::Test {
public:

};
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);

    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_2) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f.c_str());
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);

    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_3) {

    auto array = NDArrayFactory::string({3, 2}, {"alpha", "beta", "gamma", "phi", "theta", "omega"});
    
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_4) {

    NDArray array( { 3, 2 }, std::vector<const char32_t*>{ U"alpha", U"beta", U"gamma€한", U"pÿqwe", U"ß水𝄋", U"omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_5) {

    NDArray array( { 3, 2 }, std::vector<const char16_t*>{ u"alpha", u"beta", u"gamma€한", u"pÿqwe", u"ß水𝄋", u"omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_6) {

    NDArray array( { 3, 2 }, std::vector<const char*>{ "alpha", "beta", "gamma€한", "pÿqwe", "ß水𝄋", "omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_7) {

    NDArray array( { 3, 2 }, std::vector<std::u32string>{ U"alpha", U"beta", U"gamma€한", U"pÿqwe", U"ß水𝄋", U"omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_8) {

    NDArray array( { 3, 2 }, std::vector<std::u16string>{ u"alpha", u"beta", u"gamma€한", u"pÿqwe", u"ß水𝄋", u"omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_9) {

    NDArray array( { 3, 2 }, std::vector<std::string>{ "alpha", "beta", "gamma€한", "pÿqwe", "ß水𝄋", "omega" });
    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_10) {

    NDArray array(std::u32string(U"gamma€한"));
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_11) {

    NDArray array(U"gamma€한");
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_12) {

    NDArray array(std::u16string(u"gamma€한"));
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_13) {

    NDArray array(u"gamma€한");
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_14) {

    NDArray array(std::string("gamma€한"));
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_15) {

    NDArray array("gamma€한");
    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_16) {

    auto array = NDArrayFactory::string( { 3, 2 }, std::vector<std::string>{ "alpha", "beta", "gamma", "phi", "theta", "omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_17) {

    auto array = NDArrayFactory::string({ 3, 2 }, std::vector<const char*>{ "alpha", "beta", "gamma", "phi", "theta", "omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_18) {

    auto array = NDArrayFactory::string({ 3, 2 }, std::vector<std::u16string>{ u"alpha", u"beta", u"gamma", u"phi", u"theta", u"omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_19) {

    auto array = NDArrayFactory::string( { 3, 2 }, std::vector<const char16_t*>{ u"alpha", u"beta", u"gamma", u"phi", u"theta", u"omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_20) {

    auto array = NDArrayFactory::string( { 3, 2 }, std::vector<std::u32string>{ U"alpha", U"beta", U"gamma", U"phi", U"theta", U"omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_21) {

    auto array = NDArrayFactory::string( { 3, 2 }, std::vector<const char32_t*>{ U"alpha", U"òèçùà12345¤z", U"ß水𝄋ÿ€한𐍈®кею90ощъ]ї", U"phi", U"theta", U"omega" });

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_22) {
    std::u16string f(u"ß水𝄋ÿ€한𐍈®кею90ощъ]ї");
    auto array = NDArrayFactory::string(f.c_str());
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::u16string>(0);

    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_23) {
    std::u32string f(U"ß水𝄋ÿ€한𐍈®кею90ощъ]ї");
    auto array = NDArrayFactory::string(f.c_str());
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::u32string>(0);

    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_1) {
    auto array = NDArrayFactory::string( {3}, {"alpha", "beta", "gamma"});
    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_dup_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto dup = new NDArray(array.dup());

    auto z0 = array.e<std::string>(0);
    auto z1 = dup->e<std::string>(0);

    ASSERT_EQ(f, z0);
    ASSERT_EQ(f, z1);

    delete dup;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, byte_length_test_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);

    ASSERT_EQ(f.length(), StringUtils::byteLength(array));
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, byte_length_test_2) {
    auto array = NDArrayFactory::string( {2}, {"alpha", "beta"});

    ASSERT_EQ(9, StringUtils::byteLength(array));
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_split_1) {
    auto split = StringUtils::split("alpha beta gamma", " ");

    ASSERT_EQ(3, split.size());
    ASSERT_EQ(std::string("alpha"), split[0]);
    ASSERT_EQ(std::string("beta"), split[1]);
    ASSERT_EQ(std::string("gamma"), split[2]);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf8_utf16) {

    std::string utf8 = u8"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";
    std::u16string utf16Exp = u"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";

    std::u16string utf16Res;
    ASSERT_TRUE(StringUtils::u8StringToU16String(utf8, utf16Res));

    ASSERT_EQ(utf16Res.size(), utf16Exp.size());
    for (auto i = 0; i < utf16Exp.size(); i++) {
        ASSERT_EQ(utf16Exp[i], utf16Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf8_utf32) {

    std::string utf8 = u8"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";
    std::u32string utf32Exp = U"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";

    std::u32string utf32Res;
    ASSERT_TRUE(StringUtils::u8StringToU32String(utf8, utf32Res));

    ASSERT_EQ(utf32Res.size(), utf32Exp.size());
    for (auto i = 0; i < utf32Exp.size(); i++) {
        ASSERT_EQ(utf32Exp[i], utf32Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf16_utf8) {

    std::string utf8Exp = u8"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";
    std::u16string utf16 = u"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";

    std::string utf8Res;
    ASSERT_TRUE(StringUtils::u16StringToU8String(utf16, utf8Res));

    ASSERT_EQ(utf8Res.size(), utf8Exp.size());
    for (auto i = 0; i < utf8Exp.size(); i++) {
        ASSERT_EQ(utf8Exp[i], utf8Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf32_utf8) {

    std::string utf8Exp = u8"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею 90ощъ]їїщkk1q\n\t\rop~";
    std::u32string utf32 = U"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею 90ощъ]їїщkk1q\n\t\rop~";

    std::string utf8Res;
    ASSERT_TRUE(StringUtils::u32StringToU8String(utf32, utf8Res));

    ASSERT_EQ(utf8Res.size(), utf8Exp.size());
    for (auto i = 0; i < utf8Exp.size(); i++) {
        ASSERT_EQ(utf8Exp[i], utf8Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf16_utf32) {

    std::u32string utf32Exp = U"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";
    std::u16string utf16 = u"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";

    std::u32string utf32Res;
    ASSERT_TRUE(StringUtils::u16StringToU32String(utf16, utf32Res));

    ASSERT_EQ(utf32Res.size(), utf32Exp.size());
    for (auto i = 0; i < utf32Exp.size(); i++) {
        ASSERT_EQ(utf32Exp[i], utf32Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, test_unicode_utf32_utf16) {

    std::u16string utf16Exp = u"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";
    std::u32string utf32 = U"\nòèçùà12345¤zß水𝄋ÿ€한𐍈®кею90ощъ]їїщkk1q\n\t\rop~";

    std::u16string utf16Res;
    ASSERT_TRUE(StringUtils::u32StringToU16String(utf32, utf16Res));

    ASSERT_EQ(utf16Res.size(), utf16Exp.size());
    for (auto i = 0; i < utf16Exp.size(); i++) {
        ASSERT_EQ(utf16Exp[i], utf16Res[i]);
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, byte_length_test_Default) {
    
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);

    ASSERT_EQ(f.length(), StringUtils::byteLength(array));

    std::u16string f16(u"alpha");
    auto array16 = NDArrayFactory::string(f16);
    
    ASSERT_EQ(sizeof(char16_t)*f16.length(), StringUtils::byteLength(array16));

    std::u32string f32(U"alpha");
    auto array32 = NDArrayFactory::string(f32);

    ASSERT_EQ(sizeof(char32_t) * f32.length(), StringUtils::byteLength(array32));
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, byte_length_test_UTF16) {
    std::string f(u8"alpha");
    auto array = NDArrayFactory::string(f, sd::DataType::UTF16);

    ASSERT_EQ(sizeof(char16_t) * f.length(), StringUtils::byteLength(array));

    std::u16string f16(u"alpha");
    auto array16 = NDArrayFactory::string(f16, sd::DataType::UTF16);

    ASSERT_EQ(sizeof(char16_t) * f16.length(), StringUtils::byteLength(array16));

    std::u32string f32(U"alpha");
    auto array32 = NDArrayFactory::string(f32, sd::DataType::UTF16);

    ASSERT_EQ(sizeof(char16_t) * f32.length(), StringUtils::byteLength(array32));
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF16toU8) {

    std::u16string f16(u"alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f16, sd::DataType::UTF8);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);
   
    std::string f(u8"alpha水𝄋ÿ€한𐍈®кею");
    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF32toU8) {
    std::u32string f32(U"alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f32.c_str(), sd::DataType::UTF8);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);
    std::string f(u8"alpha水𝄋ÿ€한𐍈®кею");
    ASSERT_EQ(f, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF16toU16) {

    std::u16string f16(u"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f16, sd::DataType::UTF16);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    auto z = array.e<std::u16string>(0);
    
    ASSERT_EQ(z, f16);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF32toU16) {

    std::u32string f32(U"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f32, sd::DataType::UTF16);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    auto z = array.e<std::u16string>(0);
    std::u16string f16(u"€alpha水𝄋ÿ€한𐍈®кею");
    ASSERT_EQ(z, f16);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF16toU32) {

    std::u16string f16(u"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f16, sd::DataType::UTF32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    
    auto z = array.e<std::u32string>(0);
    std::u32string fres(U"€alpha水𝄋ÿ€한𐍈®кею");
    ASSERT_EQ(z, fres);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF32toU32) {

    std::u32string f32(U"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    auto z = array.e<std::u32string>(0);
    ASSERT_EQ(f32, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_UTF8toU32) {

    std::string f(u8"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f, sd::DataType::UTF32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    std::u32string f32(U"€alpha水𝄋ÿ€한𐍈®кею");
    auto z = array.e<std::u32string>(0);
    ASSERT_EQ(f32, z);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU8toUTF16) {
    auto array = NDArrayFactory::string({ 3, 2 }, { "alpha€", "beta", "gamma水", "phi", "theta", "omega水" }, sd::DataType::UTF16);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU8toUTF32) {
    auto array = NDArrayFactory::string( { 3, 2 }, { "alpha€", "beta水", "gamma", "phi", "theta", "omega" }, sd::DataType::UTF32);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U8toUTF16) {
    auto array = NDArrayFactory::string({ 3 }, { "alpha", "beta", "gamma" }, sd::DataType::UTF16);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U8toUTF32) {
    auto array = NDArrayFactory::string({ 3 }, { "alpha", "beta", "gamma" }, sd::DataType::UTF32);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU16toUTF16) {
    auto array = NDArrayFactory::string({ 3, 2 }, { u"alpha水", u"beta", u"gamma", u"phi", u"theta水", u"omega" }, sd::DataType::UTF16);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU16toUTF32) {
    auto array = NDArrayFactory::string( { 3, 2 }, { u"alpha水", u"beta", u"gamma水", u"phi", u"theta", u"omega" }, sd::DataType::UTF32);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU16toUTF8) {
    auto array = NDArrayFactory::string( { 3, 2 }, { u"alpha€", u"beta水", u"gamma", u"phi水", u"theta", u"omega" }, sd::DataType::UTF8);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U16toUTF8) {
    auto array = NDArrayFactory::string( { 3 }, { u"alpha", u"beta", u"gamma" }, sd::DataType::UTF8);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U16toUTF16) {
    auto array = NDArrayFactory::string( { 3 }, { u"alpha", u"beta", u"gamma" }, sd::DataType::UTF16);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U16toUTF32) {
    auto array = NDArrayFactory::string( { 3 }, { u"alpha水", u"beta", u"gamma水" }, sd::DataType::UTF32);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU32toUTF32) {
    auto array = NDArrayFactory::string( { 3, 2 }, { U"alpha€", U"beta水", U"gamma", U"phi", U"theta", U"omega水" }, sd::DataType::UTF32);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU32toUTF16) {
    auto array = NDArrayFactory::string({ 3, 2 }, { U"alpha水", U"水beta", U"gamma", U"phi水", U"theta", U"omega" }, sd::DataType::UTF16);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");

    printf("Array elements size: \n");
    for (int e = 0; e < array.lengthOf(); e++) {
        printf("Element %d size: %d\n", e, static_cast<int>(array.e<std::u16string>(e).size()));
    }
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_Test_StringVecU32toUTF8) {
    auto array = NDArrayFactory::string( { 3, 2 }, { U"alpha水", U"beta", U"gamma水", U"phi", U"theta", U"omega" }, sd::DataType::UTF8);

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U32toUTF32) {
    auto array = NDArrayFactory::string( { 3 }, { U"alpha", U"beta", U"gamma" }, sd::DataType::UTF32);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U32toUTF16) {
    auto array = NDArrayFactory::string( { 3 }, { U"alpha", U"beta水", U"gamma水" }, sd::DataType::UTF16);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Export_Test_U32toUTF8) {
    auto array = NDArrayFactory::string( { 3 }, { U"alpha", U"beta", U"gamma水" }, sd::DataType::UTF8);

    auto vector = array.asByteVector();
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_dup_UTF16) {
    std::u16string f(u"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto dup = new NDArray(array.dup());

    auto z0 = array.e<std::u16string>(0);
    auto z1 = dup->e<std::u16string>(0);

    ASSERT_EQ(f, z0);
    ASSERT_EQ(f, z1);

    delete dup;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_dup_UTF32) {
    std::u32string f(U"€alpha水𝄋ÿ€한𐍈®кею");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto dup = new NDArray(array.dup());

    auto z0 = array.e<std::u32string>(0);
    auto z1 = dup->e<std::u32string>(0);

    ASSERT_EQ(f, z0);
    ASSERT_EQ(f, z1);

    delete dup;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF32toUTF8) {
    
    std::u32string u32(U"€alpha水𝄋ÿ€한𐍈®кею");
    
    std::string u8(u8"€alpha水𝄋ÿ€한𐍈®кею");
    
    auto array = NDArrayFactory::string(u32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast =  array.cast(sd::DataType::UTF8);

    auto z0 = array.e<std::u32string>(0);
    auto z1 = aCast.e<std::string>(0);

    ASSERT_EQ(u32, z0);
    ASSERT_EQ(u8, z1);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF32toUTF16) {

    std::u32string u32(U"€alpha水𝄋ÿ€한𐍈®кею");

    std::u16string u16(u"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());
    
    auto aCast = array.cast(sd::DataType::UTF16);

    auto z0 = array.e<std::u32string>(0);
    auto z1 = aCast.e<std::u16string>(0);

    ASSERT_EQ(u32, z0);
    ASSERT_EQ(u16, z1);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF32toUTF32) {

    std::u32string u32(U"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u32);
    ASSERT_EQ(sd::DataType::UTF32, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF32);

    auto z0 = array.e<std::u32string>(0);
    auto z1 = aCast.e<std::u32string>(0);

    ASSERT_EQ(u32, z0);
    ASSERT_EQ(u32, z1);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF16toUTF16) {

    std::u16string u16(u"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u16);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF16);

    auto z0 = array.e<std::u16string>(0);
    auto z1 = aCast.e<std::u16string>(0);

    ASSERT_EQ(u16, z0);
    ASSERT_EQ(u16, z1);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF16toUTF32) {

    std::u32string u32(U"€alpha水𝄋ÿ€한𐍈®кею");

    std::u16string u16(u"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u16);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF32);

    auto z0 = array.e<std::u16string>(0);
    auto z1 = aCast.e<std::u32string>(0);

    ASSERT_EQ(u32, z1);
    ASSERT_EQ(u16, z0);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF16toUTF8) {

    std::string u8(u8"€alpha水𝄋ÿ€한𐍈®кею");

    std::u16string u16(u"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u16);
    ASSERT_EQ(sd::DataType::UTF16, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF8);

    auto z0 = array.e<std::u16string>(0);
    auto z1 = aCast.e<std::string>(0);

    ASSERT_EQ(u8, z1);
    ASSERT_EQ(u16, z0);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF8toUTF8) {

    std::string u8("€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u8);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF8);

    auto z0 = array.e<std::string>(0);
    auto z1 = aCast.e<std::string>(0);

    ASSERT_EQ(u8, z1);
    ASSERT_EQ(u8, z0);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF8toUTF16) {

    std::string u8(u8"€alpha水𝄋ÿ€한𐍈®кею");

    std::u16string u16(u"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u8);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF16);

    auto z0 = array.e<std::string>(0);
    auto z1 = aCast.e<std::u16string>(0);

    ASSERT_EQ(u8, z0);
    ASSERT_EQ(u16, z1);
}
/////////////////////////////////////////////////////////////////////////
TEST_F(StringTests, Basic_cast_UTF8toUTF32) {

    std::string u8(u8"€alpha水𝄋ÿ€한𐍈®кею");

    std::u32string u32(U"€alpha水𝄋ÿ€한𐍈®кею");

    auto array = NDArrayFactory::string(u8);
    ASSERT_EQ(sd::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto aCast = array.cast(sd::DataType::UTF32);

    auto z0 = array.e<std::string>(0);
    auto z1 = aCast.e<std::u32string>(0);

    ASSERT_EQ(u8, z0);
    ASSERT_EQ(u32, z1);
}

TEST_F(StringTests, test_bit_string_1) {
  // check bits -> vector conversion first
  auto vec = BitwiseUtils::valueBits(1);

  // check bits -> string conversion next;
  auto str = StringUtils::bitsToString(1);
  ASSERT_EQ(32, str.length());
  ASSERT_EQ(std::string("00000000000000000000000000000001"), str);
}