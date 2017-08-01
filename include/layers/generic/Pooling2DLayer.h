//
// @author raver119@gmail.com
//

#ifndef PROJECT_POOLING2DLAYER_H
#define PROJECT_POOLING2DLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
    namespace layers {

        // FIXME: we don't need activation function here
        template<typename T, typename AF>
        class Pooling2DLayer: public BaseLayer<T, AF> {

            void feedForward() {
                // to be implemented
            }

            void backPropagate() {
                // to be implemented
            }
        };
    }
}

#endif //PROJECT_POOLING2DLAYER_H
