//
// @author raver119@gmail.com
//

#ifndef PROJECT_DENSE_H
#define PROJECT_DENSE_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
    namespace layers {

        template<typename T>
        class DenseLayer: public BaseLayer<T> {

            void feedForward() {
                // do wxa+b here or something else
            }

            void backPropagate() {
                //
            }
        };
    }
}

#endif //PROJECT_DENSE_H
