//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

// this op will probably never become GPU-compatible
namespace nd4j {
    namespace ops {
        OP_IMPL(listdiff, 2, 2, false) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);

            std::vector<T> saved;
            std::vector<T> indices;

            for (int e = 0; e < values->lengthOf(); e++) {
                T v = values->getScalar(e);
                T extras[] = {v, 0.0f, 10.0f};
                auto idx = keep->template indexReduceNumber<simdOps::FirstIndex<T>>(extras);
                if (idx < 0) {
                    saved.emplace_back(v);
                    indices.emplace_back(e);
                }
            }

            // FIXME: we need 0-size NDArrays
            if (saved.size() == 0) {
                REQUIRE_TRUE(false, 0, "ListDiff: search returned no results");
            } else {
                auto z0 = new NDArray<T>('c', {1, (int) saved.size()});
                auto z1 = new NDArray<T>('c', {1, (int) saved.size()});

                memcpy(z0->buffer(), saved.data(), saved.size() * sizeof(T));
                memcpy(z1->buffer(), indices.data(), indices.size() * sizeof(T));

                OVERWRITE_2_RESULTS(z0, z1);
            }

            return ND4J_STATUS_OK;
        };
    }
}