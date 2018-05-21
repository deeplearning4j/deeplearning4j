//
// Created by agibsonccc on 3/30/17.
//

#include "testinclude.h"

class FileTest : public testing::Test {

};

class LoadFromStringTest :  public testing::Test {

};
/*
TEST_F(FileTest,T) {
    cnpy::NpyArray npy = cnpy::npyLoad(std::string("/home/agibsonccc/code/libnd4j/test.npy"));
    ASSERT_FALSE(npy.fortranOrder);

    ASSERT_EQ(2,npy.shape[0]);
    ASSERT_EQ(2,npy.shape[1]);
}

TEST_F(LoadFromStringTest,PathTest) {
    char *loaded = cnpy::loadFile("/home/agibsonccc/code/libnd4j/test.npy");
    cnpy::NpyArray loadedArr = cnpy::loadNpyFromPointer(loaded);
    ASSERT_FALSE(loadedArr.fortranOrder);
    ASSERT_EQ(2,loadedArr.shape[0]);
    ASSERT_EQ(2,loadedArr.shape[1]);
    double *data = reinterpret_cast<double *>(loadedArr.data);
    ASSERT_EQ(1.0,data[0]);
    ASSERT_EQ(2.0,data[1]);
    ASSERT_EQ(3.0,data[2]);
    ASSERT_EQ(4.0,data[3]);
    Nd4jPointer  pointer = reinterpret_cast<Nd4jPointer >(&loadedArr);
    int *shapeBuffer = shape::shapeBufferOfNpy(loadedArr);
    NativeOps nativeOps;
    Nd4jPointer  pointer1 = nativeOps.dataPointForNumpy(loaded);
    delete[] shapeBuffer;

    double *data2 = reinterpret_cast<double *>(pointer1);
    delete[] loaded;
}

*/