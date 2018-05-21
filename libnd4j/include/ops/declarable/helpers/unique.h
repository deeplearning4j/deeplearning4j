//
//  @author sgazeos@gmail.com
//

#ifndef __UNIQUE_H_HELPERS__
#define __UNIQUE_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int uniqueCount(NDArray<T>* input);
    template <typename T>
    int uniqueFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indices, NDArray<T>* counts);

}
}
}
#endif
