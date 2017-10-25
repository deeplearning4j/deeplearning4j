//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

        CUSTOM_OP_IMPL(strided_slice, 1, 1, true, 0, -1) {

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(strided_slice) {

            return new ShapeList();
        }
    }
}