//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONSEXECUTIONER_H
#define PROJECT_ACTIVATIONSEXECUTIONER_H

#include <layers/activations.h>

template <typename T>
class GenericActivationsExecutioner: public ActivationsExecutioner {
public:


    /**
     * This method should be backend-specific, and should be implemented accordingly
     *
     * @tparam Activation
     * @param input
     * @param inputShapeInfo
     */
    template<typename Activation>
    static inline void executeFF(T * input, int *inputShapeInfo) {
        // add special invocation here, like softmax case etc

        Nd4jIndex n = shape::length(inputShapeInfo);

//#pragma omp parallel for
        for (Nd4jIndex e = 0; e < n; e++) {
            Activation::ffActivation(input[e]);
        }
    }

    template<typename Activation>
    static inline void executeBP(T * input, T *epsilon, int *inputShapeInfo) {
        // add special invocation here, like softmax case etc

        Nd4jIndex n = shape::length(inputShapeInfo);

//#pragma omp parallel for
        for (Nd4jIndex e = 0; e < n; e++) {
            Activation::bpActivation(input[e], epsilon[e]);
        }
    }
};

#endif //PROJECT_ACTIVATIONSEXECUTIONER_H
