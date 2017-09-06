//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONS_H
#define PROJECT_ACTIVATIONS_H

#define ACTIVATIONS \
        (0, nd4j::activations::Identity) ,\
        (1, nd4j::activations::ReLU)


        /*
         * We don't really need this Executioner class, and it will be removed in favor of backend-specific helper.
         * There's just nothing to override here.
         */
namespace nd4j {
namespace activations {


template <typename T> class IActivationsExecutioner {
    public:
        // This method should be backend-specific, and should be implemented accordingly
        template<typename Activation> static void executeFF(NDArray<T> *input, NDArray<T> *output) {
            // platform-specific invocation loop here
        }

        template<typename Activation> static void executeBP(NDArray<T> *input, NDArray<T> *epsilon, NDArray<T> *output) {
            // platform-specific invocation loop here
        };
    };

// end of namespace brackets
}
}

#endif //PROJECT_ACTIVATIONS_H
