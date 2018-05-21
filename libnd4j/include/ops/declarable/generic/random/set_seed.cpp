//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_set_seed)

#include <ops/declarable/CustomOperations.h>
#include <NativeOps.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(set_seed, -2, 1, false, 0, -2) {
            REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RNG should be defined in Graph");
            auto rng = block.getRNG();
            Nd4jLong seed = 0;
            if (block.getIArguments()->size() > 0) {
                seed = INT_ARG(0);
            } else if (block.width() > 0) {
                auto input = INPUT_VARIABLE(0);
                REQUIRE_TRUE(input->isScalar(),0 ,"SetSeed: Seed operand should be scalar");
                seed = (Nd4jLong) input->getScalar(0);
            } else {
                REQUIRE_TRUE(false, 0, "SetSeed: either IArg or scalr input should be provided");
            }

            // FIXME: this approach isn't really good for cuda, since it'll assume that CUDA might get nullptr instead of stream
            NativeOps nativeOps;
            nativeOps.refreshBuffer(nullptr, seed, (Nd4jPointer) rng);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(set_seed) {
            Nd4jLong *newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);

            newshape[0] = 2;
            newshape[1] = 1;
            newshape[2] = 1;
            newshape[3] = 1;
            newshape[4] = 1;
            newshape[5] = 0;
            newshape[6] = 1;
            newshape[7] = 99;

            return SHAPELIST(newshape);
        }
    }
}

#endif