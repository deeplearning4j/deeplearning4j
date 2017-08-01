//
// @author raver119@gmail.com
//

#ifndef PROJECT_CONVOLUTIONLAYER_H
#define PROJECT_CONVOLUTIONLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
    namespace layers {

        template<typename T>
        class ConvolutionLayer: public BaseLayer<T> {

            void feedForward() {
                // im2col, wxa+b here and other fancy stuff here
            }

            void backPropagate() {
                //
            }
        };
    }
}

#endif //PROJECT_CONVOLUTIONLAYER_H
