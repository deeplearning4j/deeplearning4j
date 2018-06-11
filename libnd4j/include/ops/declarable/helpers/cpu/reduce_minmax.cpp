//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void minMaxReduceFunctor(NDArray<T>* input, NDArray<T>* gradOut, NDArray<T>* tempVals, NDArray<T>* output, bool normalize) {
            if (tempVals->isScalar()) {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    T compared = (normalize?nd4j::math::nd4j_abs((*input)(e)):(*input)(e));
                    if (nd4j::math::nd4j_abs((*tempVals)(0) - compared) < T(1.E-5f)) { // if input value equals to max
                         (*output)(e) = (normalize?(*gradOut)(0) * nd4j::math::nd4j_sign((*input)(e)):(*gradOut)(0));
                    }
                }
            }
            else {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    for (Nd4jLong j = 0; j < tempVals->lengthOf(); j++) {
                        T compared = (normalize?nd4j::math::nd4j_abs((*input)(e)):(*input)(e));
                        if (nd4j::math::nd4j_abs((*tempVals)(j) - compared) < T(1.E-5f))  // if input value equals to max
                            (*output)(e) = (normalize?(*gradOut)(j) * nd4j::math::nd4j_sign((*input)(e)):(*gradOut)(j));
                    }
                }
            }

    }

    template void minMaxReduceFunctor(NDArray<float>* input, NDArray<float>* gradOut, NDArray<float>* tempVals, NDArray<float>* output, bool normalize);
    template void minMaxReduceFunctor(NDArray<float16>* input, NDArray<float16>* gradOut, NDArray<float16>* tempVals, NDArray<float16>*  output, bool normalize);
    template void minMaxReduceFunctor(NDArray<double>* input, NDArray<double>* gradOut, NDArray<double>* tempVals, NDArray<double>* output, bool normalize);
}
}
}