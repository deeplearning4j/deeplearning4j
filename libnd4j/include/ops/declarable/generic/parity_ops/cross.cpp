//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cross)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/cross.h>

namespace nd4j {
namespace ops {
    OP_IMPL(cross, 2, 1, false) {
        auto a = INPUT_VARIABLE(0);
        auto b = INPUT_VARIABLE(1);

        REQUIRE_TRUE(a->lengthOf() == b->lengthOf(), 0, "Cross: A and B lengths should match");
        REQUIRE_TRUE(a->rankOf() >= 1 && b->rankOf() >= 1, 0, "Cross: A and B should have rank >= 1");

        // TODO: we might want to lift this restriction
        REQUIRE_TRUE(a->isSameShape(b),0, "Cross: A and B should have equal shape");
        REQUIRE_TRUE(a->sizeAt(-1) == 3, 0, "Cross: outer dimension of A and B should be equal to 3");

        auto o = OUTPUT_VARIABLE(0);

        if (a->lengthOf() == 3) {
            helpers::_cross(a, b, o);
        } else {
            helpers::_crossBatched(a, b, o);
        }

        return ND4J_STATUS_OK;
    }
}
}

#endif