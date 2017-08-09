//
// @author raver119@gmail.com
//

#ifndef PROJECT_LSTMLAYER_H
#define PROJECT_LSTMLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
namespace layers {

template<typename T, typename AF> class LSTMLayer: public BaseLayer<T, AF> {
           
    public:
        virtual int feedForward() {
            // to be implemented
        }

        virtual int backPropagate() {
            // to be implemented
        }
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////



// end of namespace brackets
}
}

#endif //PROJECT_POOLING2DLAYER_H
