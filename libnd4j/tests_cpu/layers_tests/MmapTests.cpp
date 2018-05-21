//
// Created by raver on 5/13/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <fstream>

using namespace nd4j;
using namespace nd4j::graph;

class MmapTests : public testing::Test {
public:

};

TEST_F(MmapTests, Test_Basic_Mmap_1) {
    NativeOps nativeOps;

    // just 10GB
    Nd4jLong size = 100000L;

    std::ofstream ofs("file", std::ios::binary | std::ios::out);
    ofs.seekp(size + 1024L);
    ofs.write("", 1);
    ofs.close();

    auto result = nativeOps.mmapFile(nullptr, "file", size);

    ASSERT_FALSE(result == nullptr);

    nativeOps.munmapFile(nullptr, result, size);

    remove("file");
}