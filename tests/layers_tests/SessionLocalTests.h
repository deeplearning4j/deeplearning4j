//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SESSIONLOCALTESTS_H
#define LIBND4J_SESSIONLOCALTESTS_H

#include "testlayers.h"
#include <graph/SessionLocalStorage.h>

using namespace nd4j::graph;

class SessionLocalTests : public testing::Test {
public:

};

TEST_F(SessionLocalTests, BasicTests_1) {
    VariableSpace<float> variableSpace;
    SessionLocalStorage<float> storage(&variableSpace, nullptr);

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.startSession();
    }

    ASSERT_EQ(4, storage.numberOfSessions());

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.endSession();
    }

    ASSERT_EQ(0, storage.numberOfSessions());
}


TEST_F(SessionLocalTests, BasicTests_2) {
    VariableSpace<float> variableSpace;
    SessionLocalStorage<float> storage(&variableSpace, nullptr);
    auto alpha = new nd4j::NDArray<float>(5,5,'c');
    alpha->assign(0.0);

    variableSpace.putVariable(-1, alpha);

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.startSession();

        auto varSpace = storage.localVariableSpace();

        auto arr = varSpace->getVariable(-1)->getNDArray();
        arr->applyScalar<simdOps::Add<float>>((float) e+1);
    }
}

#endif //LIBND4J_SESSIONLOCALTESTS_H
