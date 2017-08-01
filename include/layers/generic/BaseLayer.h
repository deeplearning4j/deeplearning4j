//
// @author raver119@gmail.com
//

#ifndef PROJECT_BASELAYER_H
#define PROJECT_BASELAYER_H


#include <layers/layers.h>
#include <layers/generic/ActivationsExecutioner.h>


namespace nd4j {
    namespace layers {

        template<typename T, typename AF>
        class BaseLayer: public INativeLayer<T> {

        public:
            T *allocate(long bytes) {
                this->allocated += bytes;

                return (T *) (this->workspace + bytes);
            }


            /*
             * TODO: we need platform-specific RNG here (!!!)
             */

            void dropOutHelper(T *input, int *shapeInfo) {
                // basically we loop over input here, and we're using inverted dropout here

                // we probably should allocate temp array here, and replace input pointer
            }

            void dropConnectHelper(T *input, int *shapeInfo) {
                // and here we just loop over copy of params for dropout. regular dropout is use

                // we probably should allocate temp array here, and replace params pointer
            }
        };
    }
}

#endif //PROJECT_BASELAYER_H
