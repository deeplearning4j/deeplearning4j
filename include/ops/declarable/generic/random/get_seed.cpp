//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(get_seed, -2, 1, false, 0, 0) {
            REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RNG should be defined in Graph");
            auto rng = block.getRNG();
            auto z = OUTPUT_VARIABLE(0);

            z->putScalar(0, (T) rng->getSeed());

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(get_seed) {
            int *newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(2), int);

            newshape[0] = 2;
            newshape[1] = 1;
            newshape[2] = 1;
            newshape[3] = 1;
            newshape[4] = 1;
            newshape[5] = 0;
            newshape[6] = 1;
            newshape[7] = 99;

            return new ShapeList(newshape);
        }
    }
}