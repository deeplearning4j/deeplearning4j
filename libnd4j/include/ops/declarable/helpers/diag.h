//
//  @author GS <sgazeos@gmail.com>
//
#ifndef __DIAG_H_HELPERS__
#define __DIAG_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void diagFunctor(NDArray<T> const* input, NDArray<T>* output);

}
}
}
#endif
