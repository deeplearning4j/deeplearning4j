//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/shape.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(tile_to_shape, 1, 1, true, 0, -1) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        if (block.isInplace()) {
            input->tileToShape(*block.getIArguments());
        } else {
            input->tileToShape(*block.getIArguments(), output);
        }

        return Status::OK();
    }

    DECLARE_SHAPE_FN(tile_to_shape) {
        auto in = inputShape->at(0);

        // output shape always equals to arguments
        int *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(block.numI()), int);

        if (shape::order(in) == 'c')
            shape::shapeBuffer(block.numI(), block.getIArguments()->data(), newShape);
        else 
            shape::shapeBufferFortran(block.numI(), block.getIArguments()->data(), newShape);

        return new ShapeList(newShape);
    }
}
}