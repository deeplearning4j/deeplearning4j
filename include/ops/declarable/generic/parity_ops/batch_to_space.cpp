//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(batch_to_space, 3, 1, false, 0, -2) {

        return Status::THROW("Not implemented yet");
    }

    DECLARE_SHAPE_FN(batch_to_space) {
        return new ShapeList();
    }
}
}