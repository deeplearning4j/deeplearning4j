//
//  @author GS <sgazeos@gmail.com>
//
#ifndef __MATRIX_DIAG_HELPERS__
#define __MATRIX_DIAG_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int matrixDiag(NDArray<T> const* input, NDArray<T>* output);

}
}
}
#endif
