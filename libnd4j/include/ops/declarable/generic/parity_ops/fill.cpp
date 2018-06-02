//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_fill)

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
    namespace ops {
        
        CUSTOM_OP_IMPL(fill, 1, 1, false, 1, 0) {
            auto shapeArray = INPUT_VARIABLE(0);
            
            T scalar = T_ARG(0);             

            std::vector<Nd4jLong> shape((int) shapeArray->lengthOf());

            for (int e = 0; e < shapeArray->lengthOf(); e++)
                shape[e] = static_cast<Nd4jLong>((*shapeArray)(e));

            auto result = NDArrayFactory<T>::valueOf(shape, scalar, 'c');

            OVERWRITE_RESULT(result);            

            return ND4J_STATUS_OK;
        };

        
        DECLARE_SHAPE_FN(fill) {

            auto shapeArray = INPUT_VARIABLE(0);

            const int len = shapeArray->lengthOf();
            Nd4jLong *newShape = nullptr;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(len), Nd4jLong);            

            newShape[0] = len;
            for (int e = 0; e < shapeArray->lengthOf(); e++)
                newShape[e+1] = static_cast<Nd4jLong>((*shapeArray)(e));
            
            shape::updateStrides(newShape, 'c');

            return SHAPELIST(newShape);
        };
    }
}

#endif