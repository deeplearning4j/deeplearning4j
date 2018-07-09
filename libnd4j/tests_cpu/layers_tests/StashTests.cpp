//
// @author raver119@gmail.com
//

#ifndef LIBND4J_STASHTESTS_H
#define LIBND4J_STASHTESTS_H

#include <NDArray.h>
#include "testlayers.h"
#include <graph/Stash.h>

using namespace nd4j;
using namespace nd4j::graph;

class StashTests : public testing::Test {
public:

};

TEST_F(StashTests, BasicTests_1) {
    Stash<float> stash;

    auto alpha = new NDArray<float> ('c',{5, 5});
    alpha->assign(1.0);

    auto beta = new NDArray<float> ('c',{5, 5});
    beta->assign(2.0);

    auto cappa = new NDArray<float> ('c',{5, 5});
    cappa->assign(3.0);

    stash.storeArray(1, "alpha", alpha);
    stash.storeArray(2, "alpha", beta);
    stash.storeArray(3, "cappa", cappa);

    ASSERT_TRUE(stash.checkStash(1, "alpha"));
    ASSERT_TRUE(stash.checkStash(2, "alpha"));
    ASSERT_TRUE(stash.checkStash(3, "cappa"));

    ASSERT_FALSE(stash.checkStash(3, "alpha"));
    ASSERT_FALSE(stash.checkStash(2, "beta"));
    ASSERT_FALSE(stash.checkStash(1, "cappa"));
}


TEST_F(StashTests, BasicTests_2) {
    Stash<float> stash;

    auto alpha = new NDArray<float>('c',{5, 5});
    alpha->assign(1.0);

    auto beta = new NDArray<float>('c',{5, 5});
    beta->assign(2.0);

    auto cappa = new NDArray<float>('c',{5, 5});
    cappa->assign(3.0);

    stash.storeArray(1, "alpha1", alpha);
    stash.storeArray(1, "alpha2", beta);
    stash.storeArray(1, "alpha3", cappa);

    ASSERT_FALSE(stash.checkStash(2, "alpha1"));
    ASSERT_FALSE(stash.checkStash(2, "alpha2"));
    ASSERT_FALSE(stash.checkStash(2, "alpha3"));

    ASSERT_TRUE(alpha == stash.extractArray(1, "alpha1"));
    ASSERT_TRUE(beta == stash.extractArray(1, "alpha2"));
    ASSERT_TRUE(cappa == stash.extractArray(1, "alpha3"));

}

#endif //LIBND4J_STASHTESTS_H
