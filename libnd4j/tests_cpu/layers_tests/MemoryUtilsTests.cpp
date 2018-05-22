//
// Created by raver119 on 11.10.2017.
//

#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include "testlayers.h"

using namespace nd4j::memory;

class MemoryUtilsTests : public testing::Test {
public:

};

TEST_F(MemoryUtilsTests, BasicRetrieve_1) {
    MemoryReport reportA;
    MemoryReport reportB;

#ifdef _WIN32
    if (1 > 0)
        return;
#endif


    MemoryUtils::retrieveMemoryStatistics(reportA);


    ASSERT_NE(reportA, reportB);
}
