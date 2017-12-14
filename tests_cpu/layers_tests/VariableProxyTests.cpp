//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/VariableProxy.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableProxyTests : public testing::Test {
public:

};


TEST_F(VariableProxyTests, Test_Simple_1) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace<float> ref;

    ref.putVariable(119, x);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy<float> proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));
}


TEST_F(VariableProxyTests, Test_Simple_2) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace<float> ref;

    ASSERT_FALSE(ref.hasVariable(119));

    VariableProxy<float> proxy(&ref);

    ASSERT_FALSE(proxy.hasVariable(119));

    proxy.putVariable(119, x);

    ASSERT_FALSE(ref.hasVariable(119));

    ASSERT_TRUE(proxy.hasVariable(119));
}


TEST_F(VariableProxyTests, Test_Simple_3) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = new NDArray<float>('c', {2, 2}, {4, 2, 3, 1});
    VariableSpace<float> ref;

    ref.putVariable(119, x);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy<float> proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));

    proxy.putVariable(119, y);

    ASSERT_TRUE(ref.hasVariable(119));

    ASSERT_TRUE(proxy.hasVariable(119));

    auto z0 = ref.getVariable(119)->getNDArray();
    auto z1 = proxy.getVariable(119)->getNDArray();

    ASSERT_FALSE(z0 == z1);
    ASSERT_TRUE(y == z1);
    ASSERT_TRUE(x == z0);
}


TEST_F(VariableProxyTests, Test_Cast_1) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = new NDArray<float>('c', {2, 2}, {4, 2, 3, 1});
    VariableSpace<float> ref;

    ref.putVariable(-119, x);

    ASSERT_TRUE(ref.hasVariable(-119));

    VariableProxy<float> proxy(&ref);
    auto cast = (VariableSpace<float> *) &proxy;

    ASSERT_TRUE(cast->hasVariable(-119));

    cast->putVariable(-119, y);

    ASSERT_TRUE(ref.hasVariable(-119));

    ASSERT_TRUE(cast->hasVariable(-119));

    auto z0 = ref.getVariable(-119)->getNDArray();
    auto z1 = cast->getVariable(-119)->getNDArray();

    ASSERT_FALSE(z0 == z1);
    ASSERT_TRUE(y == z1);
    ASSERT_TRUE(x == z0);
}