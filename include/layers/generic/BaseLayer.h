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


        };
    }
}

#endif //PROJECT_BASELAYER_H
