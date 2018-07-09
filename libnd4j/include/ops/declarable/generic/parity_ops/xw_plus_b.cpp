//
//  xw_plus_b op. Created by GS <george@skymind.io> 31.01.2018
//
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_xw_plus_b)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>
#include <MmulHelper.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(xw_plus_b, 3, 1, false, 0, 0) {
            NDArray<T>* x = INPUT_VARIABLE(0);
            NDArray<T>* y = INPUT_VARIABLE(1);
            NDArray<T>* b = INPUT_VARIABLE(2);
            NDArray<T>* z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() <= 2 && y->rankOf() <= 2 && z->rankOf() <= 2, 0, "xw_plus_b: Input and Output NDArrays should have rank less or equal to 2");
            REQUIRE_TRUE(b->isVector() && b->lengthOf() == z->sizeAt(-1), 0, "xw_plus_b: Input vector should have proper dimension 1x%i. "
                "But %i != %i.", z->sizeAt(-1), b->lengthOf(), z->sizeAt(-1));
            // multiply x to y
            MmulHelper<T>::mmul(x, y, z, T(1.0f), T(0.0f));

            // adding b vector
            z->addiRowVector(b);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(xw_plus_b) {
            auto outputShape = ShapeUtils<T>::matrixProductShape(inputShape->at(0), inputShape->at(1), false, false, block.getWorkspace()); 
            
            return SHAPELIST(outputShape);
        }

    }
}

#endif