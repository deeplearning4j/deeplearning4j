//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_OPS_H
#define LIBND4J_RANDOM_OPS_H

#ifdef __CUDACC__
#define random_def __device__ inline static
#else
#define random_def inline static
#endif

#include <helpers/helper_random.h>

namespace randomOps {

    template<typename T>
    class BoundedDistribution {
    public:

      random_def T op(T value, T idx, RandomHelper *helper, T *extraParams) {
          return 0.0f;
      }
    };
}

#endif //LIBND4J_RANDOM_OPS_H
