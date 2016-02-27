#include <transform.h>

namespace nd4j {
    namespace functions {
        namespace transform {
/**
 * Base class with a base implementation of
 * transform
 */
            template<typename T>
            class BaseTransform : public Transform<T> {

            public:
                __device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize) {
                    int totalThreads = gridDim.x * blockDim.x;
                    int tid = threadIdx.x;
                    int i = blockIdx.x * blockDim.x + tid;
                    /* equal, positive, non-unit increments. */
                    for (; i < n; i += totalThreads) {
                        result[i * incy] = op(dy[i * incy], params);
                    }


                }
            };

        }
    }
}
