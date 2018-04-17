//
//  @author sgazeos@gmail.com
//
#ifndef __CONFUSION_H_HELPERS__
#define __CONFUSION_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void confusionFunctor(NDArray<T>* labels, NDArray<T>* predictions, NDArray<T>* weights, NDArray<T>* output);

}
}
}
#endif
