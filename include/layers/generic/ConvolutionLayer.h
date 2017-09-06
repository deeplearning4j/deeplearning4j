//
// @author raver119@gmail.com
//

#ifndef PROJECT_CONVOLUTIONLAYER_H
#define PROJECT_CONVOLUTIONLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
namespace layers {

template<typename T, typename AF> class ConvolutionLayer: public BaseLayer<T, AF> {

    public:
        virtual int feedForward() {}
        virtual int backPropagate() {} 
};



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////


// end of namespace brackets
}
}

#endif //PROJECT_CONVOLUTIONLAYER_H
