//
// @author raver119@gmail.com
//

#ifndef PROJECT_LAYERS_FACTORY_H
#define PROJECT_LAYERS_FACTORY_H

/**
 * This file is main entry point into NativeLayers
 */


#include <helpers/helper_random.h>
#include <helpers/helper_generator.h>
#include <layers/layers.h>
#include <layers/activations.h>
#include <layers/activations/ops.h>

/////////// here we should have backend-specific includes

// TODO: we want CHIP/ARCH vars being passed from build script here
#include <layers/generic/available.h>

namespace nd4j {
namespace layers {

template <typename T> class LayerFactory {

    public:
        // we should give out layer instances depending on layerNum here
        // TODO: we want to pass things like dropout, dropconnect, probabilities, whatever else here
        static INativeLayer<float>* getNewLayerFloat(int layerNum, int activationNum);
            
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////


template <typename T> INativeLayer<float>* LayerFactory<T>::getNewLayerFloat(int layerNum, int activationNum) {
    // macro required here, based on list of available layers, declared in available.h
    //return new DenseLayer<float, nd4j::activations::ReLU<float>>();

    BUILD_LAYERS_FACTORY(float, OPS_A(NATIVE_LAYERS), OPS_B(ACTIVATIONS))

    // return null here
    return nullptr;
}


// end of namespace brackets
}
}

#endif //PROJECT_LAYERS_FACTORY_H
