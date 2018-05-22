//
//  @author sgazeos@gmail.com
//
#ifndef __DROP_OUT_HELPERS__
#define __DROP_OUT_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* output, NDArray<T>* reduceShape, int seed, T probValue);

}
}
}
#endif
