//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mergeadd)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(mergeadd, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mergesum, mergeadd);
        DECLARE_SYN(add_n, mergeadd);
        DECLARE_SYN(addn, mergeadd);
        DECLARE_SYN(accumulaten, mergeadd);
        DECLARE_SYN(accumulate_n, mergeadd);
    }
}

#endif