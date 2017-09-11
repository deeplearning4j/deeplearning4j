//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONSEXECUTIONER_H
#define PROJECT_ACTIVATIONSEXECUTIONER_H

#include <layers/activations.h>

using namespace nd4j;

template <typename T> class ActivationsExecutioner {

    public: 
    
    // This method should be backend-specific, and should be implemented accordingly
    template<typename Activation> inline static void executeFF(NDArray<T> *input, NDArray<T> *output);
    
    template<typename Activation> inline static void executeBP(NDArray<T> *input, NDArray<T> *epsilon, NDArray<T> *output);
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// This method should be backend-specific, and should be implemented accordingly
template<typename T> template<typename Activation> 
void ActivationsExecutioner<T>::executeFF(NDArray<T> *input, NDArray<T> *output) {
    // add special invocation here, like softmax case etc
    if (Activation::requiresSpecialFF()) {
        Activation::ffActivation(input, output);
        return;
    }

    Nd4jIndex n = input->lengthOf();
    //#pragma omp parallel for
    for (Nd4jIndex e = 0; e < n; e++) {
       output->getBuffer()[e] = Activation::ffActivation(input->getBuffer()[e]);
    }
}


template<typename T> template<typename Activation> 
void ActivationsExecutioner<T>::executeBP(NDArray<T> * input, NDArray<T> *epsilon, NDArray<T> *output) {
        // add special invocation here, like softmax case etc
        if (Activation::requiresSpecialBP()) {
            Activation::bpActivation(input, epsilon, output);
            return;
        }

        Nd4jIndex n = input->lengthOf();
        //#pragma omp parallel for
        for (Nd4jIndex e = 0; e < n; e++) {
            output->getBuffer()[e] = Activation::bpActivation(input->getBuffer()[e], epsilon->getBuffer()[e]);
        }
}
#endif //PROJECT_ACTIVATIONSEXECUTIONER_H
