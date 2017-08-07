//
// @author raver119@gmail.com
//

#ifndef PROJECT_BASELAYER_H
#define PROJECT_BASELAYER_H


#include <layers/layers.h>
#include "../../../blas/NativeOpExcutioner.h"
#include <layers/generic/ActivationsExecutioner.h>

// FIXME: we need to use MKL/OpenBLAS/whatever here, but temporary this will work fine
#include <ops/gemm.h>

namespace nd4j {
namespace layers {

template<typename T, typename AF> class BaseLayer: public INativeLayer<T> {
    
    public:        
        // override virtual method, this method "allocates" memory chunk from workspace
        T *allocate(long bytes) {
            this->allocated += bytes;

            return (T *) (this->workspace + bytes);
        }

        // TODO: we need platform-specific RNG here (!!!)

        // basically we loop over input here, and we're using inverted dropout here
        void dropOutHelper(NDArray<T> *input);

         // and here we just loop over copy of params for dropout. regular dropout is use
        void dropConnectHelper(NDArray<T> *input);
    };


/////// inmplementation part ///////


template<typename T, typename AF> void BaseLayer<T,AF>::dropOutHelper(NDArray<T> *input) {
    // basically we loop over input here, and we're using inverted dropout here
    if (this->rng == nullptr)
        throw std::invalid_argument("RNG is undefined");

    // executing DropOutInverted here
    T *extras = new T[1] {this->pDropOut};
    NativeOpExcutioner<T>::execRandom(2, (Nd4jPointer) this->rng, input->buffer, input->shapeInfo, input->buffer, input->shapeInfo, extras);

    delete[] extras;
}


template<typename T, typename AF> void BaseLayer<T,AF>::dropConnectHelper(NDArray<T> *input) {
    // and here we just loop over copy of params for dropout. regular dropout is use
    if (this->rng == nullptr)
        throw std::invalid_argument("RNG is undefined");

    // executing regular DropOut op here for DropConnect
    T *extras = new T[1] {this->pDropConnect};
    NativeOpExcutioner<T>::execRandom(1, (Nd4jPointer) this->rng, input->buffer, input->shapeInfo, input->buffer, input->shapeInfo, extras);

    delete[] extras;
}


// end of namespace brackets
}
}

#endif //PROJECT_BASELAYER_H
