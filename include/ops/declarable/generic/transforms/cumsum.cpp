//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/prefix.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(cumsum, 1, 1, true, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (block.getIArguments()->size() == 0 && block.width() == 1) {
                // all at once case
                nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo());
            } else {
                std::vector<int> dims = *(block.getIArguments());

                nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(input, output, dims);
            }

            return ND4J_STATUS_OK;
        }
    }
}