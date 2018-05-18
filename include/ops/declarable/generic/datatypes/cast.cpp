//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cast)

#include <array/DataTypeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(cast, 1, 1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // TODO: once we add support for multiple dtypes - uncommend this
            /*
            int it = INT_ARG(0);
            DataType newType = DataTypeUtils::fromInt(it);

            input->cast(output, newType);
            */

            if (!block.isInplace())
                output->assign(input);
            
            STORE_RESULT(output);
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Cast, cast);

        DECLARE_SHAPE_FN(cast) {
            auto inShape = inputShape->at(0);

            int it = INT_ARG(0);
            DataType newType = DataTypeUtils::fromInt(it);

            // FIXME: remove memcpy here
            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            // TODO: put sign of dtype into shapeinfo?

            return SHAPELIST(newShape);
        }
    }
}

#endif