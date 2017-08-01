//
// @author raver119@gmail.com
//

#ifndef PROJECT_LAYERS_FACTORY_H
#define PROJECT_LAYERS_FACTORY_H

/**
 * This file is main entry point into NativeLayers
 */


#include <layers/layers.h>
#include <layers/activations.h>
#include <layers/activations/ops.h>

/////////// here we should have backend-specific includes

// TODO: we want CHIP/ARCH vars being passed from build script here
#include <layers/generic/available.h>

namespace nd4j {
    namespace layers {

        template <typename T>
        class LayerFactory {

        public:

            // we should give out layer instances depending on layerNum here
            // TODO: we want to pass things like dropout, dropconnect, probabilities, whatever else here
            static INativeLayer<float>* getNewLayerFloat(int layerNum) {
                // macro required here, based on list of available layers, declared in available.h
                return new DenseLayer<float, nd4j::activations::ReLU<float>>();
            }
        };
    }
}

#endif //PROJECT_LAYERS_FACTORY_H
