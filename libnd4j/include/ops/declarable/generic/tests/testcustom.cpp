//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_testcustom)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(testcustom, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

            //new NDArray<T>('c', {100, 100});

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(testcustom) {
            // this test op will just return back original shape doubled
            Nd4jLong *shapeOf;
            ALLOCATE(shapeOf, block.getWorkspace(), shape::rank(inputShape->at(0)), Nd4jLong);

            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), Nd4jLong);

            for (int e = 0; e < shape::rank(inputShape->at(0)); e++)
                shapeOf[e] = inputShape->at(0)[e+1] * 2;


            shape::shapeBuffer(shape::rank(inputShape->at(0)), shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());

            return SHAPELIST(newShape);
        }
    }
}

#endif
