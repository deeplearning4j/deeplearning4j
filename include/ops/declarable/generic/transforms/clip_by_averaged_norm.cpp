//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyavgnorm)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(clipbyavgnorm, 1, 1, true, 1, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            T clip_norm = T_ARG(0);

            if (block.getIArguments()->size() == 0) {
                // all-reduce
                T n2 = input->template reduceNumber<simdOps::Norm2<T>>() / input->lengthOf();
                if (n2 <= clip_norm) {
                    if (!block.isInplace())
                        output->assign(input);
                } else {
                    auto lambda = LAMBDA_T(_x, clip_norm, n2) {
                        return _x * clip_norm / n2;
                    };

                    input->applyLambda(lambda, output);
                }
            } else {
                // along dimension
                auto norm2 = input->template reduceAlongDims<simdOps::Norm2<T>>(*block.getIArguments(), false);

                if (!block.isInplace())
                        output->assign(input);

                auto tads = NDArrayFactory<T>::allTensorsAlongDimension(output, *block.getIArguments());
                // TODO: make this CUDA-compliant somehow
                for (int e = 0; e < tads->size(); e++) {
                    T n2 = norm2.getScalar(e) / tads->at(e)->lengthOf();

                    if (n2 > clip_norm) {
                        auto lambda = LAMBDA_T(_x, clip_norm, n2) {
                            return _x * clip_norm / n2;
                        };

                        tads->at(e)->applyLambda(lambda, output);
                    }
                }

                delete tads;
            }

            return ND4J_STATUS_OK;
        }
    }
}

#endif