//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(fill, 1, 1, false, -2, 0) {
            auto shapeArray = INPUT_VARIABLE(0);
            
            T scalar = 0;
            if (block.getTArguments()->size() > 0) {
                scalar = T_ARG(0);
            } else if (block.width() > 1) {
                auto scArr = INPUT_VARIABLE(1);
                scalar = scArr->getScalar(0);
            }

            std::vector<int> shape((int) shapeArray->lengthOf());

            for (int e = 0; e < shapeArray->lengthOf(); e++)
                shape[e] = (int) shapeArray->getScalar(e);

            auto result = NDArrayFactory<T>::valueOf(shape, scalar, 'c');

            OVERWRITE_RESULT(result);            

            return ND4J_STATUS_OK;
        };

        DECLARE_SHAPE_FN(fill) {
            // this function won't be used in practice, since this is runtime operation, so shape will be always overwritten
            auto inp = inputShape->at(0);

            int len = shape::length(inp);

            std::vector<int> shape((int) shape::length(inp));
            for (int e = 0; e < shape::length(inp); e++)
                shape[e] = 1;

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape::length(inp)), int);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            return SHAPELIST(newShape);
        };
    }
}