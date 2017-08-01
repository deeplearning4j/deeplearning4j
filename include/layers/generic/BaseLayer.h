//
// @author raver119@gmail.com
//

#ifndef PROJECT_BASELAYER_H
#define PROJECT_BASELAYER_H


#include <layers/layers.h>

namespace nd4j {
    namespace layers {

        template<typename T>
        class BaseLayer: public INativeLayer<T> {

        protected:
            T *allocate(long bytes) {
                allocated += bytes;

                return (T *) (workspace + bytes);
            }


            /*
             * TODO: we need platform-specific RNG here (!!!)
             */

            void dropOutHelper(T *input, int *shapeInfo) {
                // basically we loop over input here, and we're using inverted dropout here

            }

            void dropConnectHelper(T *input, int *shapeInfo) {
                // and here we just loop over copy of params for dropout. regular dropout is use
            }
        };
    }
}

#endif //PROJECT_BASELAYER_H
