//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <climits>
#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
#include <NDArray.h>
#include <graph/Variable.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/Context.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(merge, -1, 1, true) {
            // TODO: to be removed
            return ND4J_STATUS_OK;
        }


        OP_IMPL(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        //////////////////////////////////////////////////////////////////////////
        // test op, non-divergent
        OP_IMPL(testop2i2o, 2, 2, true) {
            //nd4j_printf("CPU op used!\n","");
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto xO = OUTPUT_VARIABLE(0);
            auto yO = OUTPUT_VARIABLE(1);

            x->template applyScalar<simdOps::Add<T>>(1.0, xO, nullptr);
            y->template applyScalar<simdOps::Add<T>>(2.0, yO, nullptr);

            STORE_2_RESULTS(*xO, *yO);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);

        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(testcustom, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

            //new NDArray<T>('c', {100, 100});

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(testcustom) {
            // this test op will just return back original shape doubled
            int *shapeOf;
            ALLOCATE(shapeOf, block.getWorkspace(), shape::rank(inputShape->at(0)), int);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);

            for (int e = 0; e < shape::rank(inputShape->at(0)); e++)
                shapeOf[e] = inputShape->at(0)[e+1] * 2;


            shape::shapeBuffer(shape::rank(inputShape->at(0)), shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());

            return new ShapeList(newShape);
        }

        REDUCTION_OP_IMPL(testreduction, 1, 1, false, 0, -1) {
            auto z = OUTPUT_VARIABLE(0);

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
    }
}

#endif //LIBND4J_PARITY_OPS_H

