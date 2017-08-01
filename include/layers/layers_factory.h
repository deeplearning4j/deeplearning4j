//
// @author raver119@gmail.com
//

#ifndef PROJECT_LAYERS_FACTORY_H
#define PROJECT_LAYERS_FACTORY_H

/**
 * This file is main entry point into NativeLayers
 */


#include <layers/layers.h>

/////////// here we should have backend-specific includes
#include <layers/generic/available.h>

namespace nd4j {
    namespace layers {

        template <typename T>
        class LayerFactory {

        public:

            // we should give out layer instances depending on layerNum here
            // TODO: we want to pass things like dropout, dropconnect, probabilities, whatever else here
            static INativeLayer* getNewLayer(int layerNum) {
                // macro required here, based on list of available layers, declared in available.h
                return new DenseLayer();
            }
        };
    }
}

#endif //PROJECT_LAYERS_FACTORY_H
