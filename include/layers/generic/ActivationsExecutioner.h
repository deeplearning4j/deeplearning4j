//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONSEXECUTIONER_H
#define PROJECT_ACTIVATIONSEXECUTIONER_H

#include <layers/activations.h>

template <typename T> class ActivationsExecutioner {

    public: 
    
    // This method should be backend-specific, and should be implemented accordingly
    template<typename Activation> inline static void executeFF(T *input, T *output, int *inputShapeInfo);
    
    template<typename Activation> inline static void executeBP(T *input, T *epsilon, T *output, int *inputShapeInfo);
};


// This method should be backend-specific, and should be implemented accordingly
template<typename T> template<typename Activation> 
void ActivationsExecutioner<T>::executeFF(T *input, T *output, int *inputShapeInfo) {
    // add special invocation here, like softmax case etc
    if (Activation::requiresSpecialFF()) {
        Activation::ffActivation(input, output, inputShapeInfo);
        return;
    }

    Nd4jIndex n = shape::length(inputShapeInfo);
    //#pragma omp parallel for
    for (Nd4jIndex e = 0; e < n; e++) {
       output[e] = Activation::ffActivation(input[e]);
    }
}


template<typename T> template<typename Activation> 
void ActivationsExecutioner<T>::executeBP(T * input, T *epsilon, T *output, int *inputShapeInfo) {
        // add special invocation here, like softmax case etc
        if (Activation::requiresSpecialBP()) {
            Activation::bpActivation(input, epsilon, output, inputShapeInfo);
            return;
        }

        Nd4jIndex n = shape::length(inputShapeInfo);
        //#pragma omp parallel for
        for (Nd4jIndex e = 0; e < n; e++) {
            output[e] = Activation::bpActivation(input[e], epsilon[e]);
        }
}
#endif //PROJECT_ACTIVATIONSEXECUTIONER_H
